"""
Unified ablation runner for the paper-facing experiments.

Design goals:
- Reuse the current V30 pipeline for all switchable modules.
- Materialize legacy V7 / V18 scripts from git history only when needed.
- Keep outputs isolated and machine-readable for later aggregation.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUTPUT_DIR = BASE_DIR / "outputs" / "ablation_suite"
MANIFEST_PATH = OUTPUT_DIR / "case_manifest.json"
MANIFEST_LOCK_PATH = OUTPUT_DIR / "case_manifest.lock"
STATUS_DIR = OUTPUT_DIR / "status"
LOG_DIR = OUTPUT_DIR / "logs"
CSV_PATH = OUTPUT_DIR / "ablation_summary.csv"
MD_PATH = OUTPUT_DIR / "ablation_summary.md"
PROGRESS_PREFIX = "__AB_PROGRESS__"


def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


def env_flag(name: str, default: str = "0") -> bool:
    return env_str(name, default).lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = env_str(name, str(default))
    return int(value)


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sanitize_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "task"


def read_last_log_line(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for line in reversed(lines):
        text = line.strip()
        if text:
            return text
    return ""


def normalize_progress_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    parts = [part.strip() for part in cleaned.split("\n")]
    for part in reversed(parts):
        if part:
            return part
    return ""


def format_progress_payload(payload: dict[str, object]) -> str:
    kind = str(payload.get("kind") or "")
    category = str(payload.get("category") or "")
    current = payload.get("current")
    total = payload.get("total")
    image = str(payload.get("image") or "")
    if kind == "category_start":
        return f"{category} start {current}/{total}"
    if kind == "resume":
        return f"{category} resume at {current}/{total}"
    if kind == "image_progress":
        suffix = f" | {image}" if image else ""
        return f"{category} {current}/{total}{suffix}"
    if kind == "category_done":
        return f"{category} done {current}/{total}"
    return json.dumps(payload, ensure_ascii=False)


def pump_child_output(
    proc: subprocess.Popen[str],
    log_path: Path,
    latest_state: dict[str, str],
) -> None:
    assert proc.stdout is not None
    latest_state.setdefault("line_buffer", "")
    with log_path.open("a", encoding="utf-8") as log_file:
        while True:
            chunk = proc.stdout.read(1)
            if chunk == "":
                break
            log_file.write(chunk)
            if chunk in {"\r", "\n"}:
                log_file.flush()
                line = latest_state.get("line_buffer", "").strip()
                latest_state["line_buffer"] = ""
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        payload = json.loads(line[len(PROGRESS_PREFIX):])
                        latest_state["latest"] = format_progress_payload(payload)
                    except Exception:
                        latest_state["latest"] = line
                elif line:
                    latest_state["latest"] = line
            else:
                latest_state["line_buffer"] = (latest_state.get("line_buffer", "") + chunk)[-2000:]
            latest_state["buffer"] = (latest_state.get("buffer", "") + chunk)[-4000:]
            candidate = normalize_progress_text(latest_state["buffer"])
            if candidate and not candidate.startswith(PROGRESS_PREFIX):
                latest_state["latest"] = candidate
        log_file.flush()


def run_cmd(
    script_path: Path,
    extra_env: dict[str, str] | None = None,
    *,
    status_label: str | None = None,
    heartbeat_sec: int = 10,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ABLATION_PROGRESS", "1")
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    label = status_label or script_path.name
    log_path = LOG_DIR / f"{sanitize_label(label)}.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n===== {now_iso()} | START {label} =====\n")
        log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )
    latest_state = {"buffer": "", "latest": ""}
    reader = threading.Thread(target=pump_child_output, args=(proc, log_path, latest_state), daemon=True)
    reader.start()
    started_at = time.time()
    last_feedback = started_at
    last_log_hint = ""
    print(f"[supervisor] {label} log={log_path}")
    while True:
        return_code = proc.poll()
        if return_code is not None:
            reader.join(timeout=2)
            elapsed = format_elapsed(time.time() - started_at)
            if return_code != 0:
                last_line = latest_state.get("latest") or read_last_log_line(log_path)
                hint = f" | last={last_line}" if last_line else ""
                raise subprocess.CalledProcessError(return_code, cmd, output=f"log={log_path}{hint}")
            final_line = latest_state.get("latest")
            hint = f" | last={final_line}" if final_line else ""
            print(f"[supervisor] {label} finished in {elapsed}{hint}")
            return
        now = time.time()
        if now - last_feedback >= heartbeat_sec:
            elapsed = format_elapsed(now - started_at)
            last_line = latest_state.get("latest") or read_last_log_line(log_path)
            hint = ""
            if last_line and last_line != last_log_hint:
                hint = f" | last={last_line}"
                last_log_hint = last_line
            print(f"[supervisor] {label} still running | elapsed={elapsed}{hint}")
            last_feedback = now
        time.sleep(5)


def run_python_module(script_path: Path, extra_env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ABLATION_PROGRESS", "1")
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def manifest_lock(timeout_sec: float = 30.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_sec
    while True:
        try:
            fd = os.open(str(MANIFEST_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out acquiring manifest lock: {MANIFEST_LOCK_PATH}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            MANIFEST_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def bank_path_current(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v30{bank_suffix}" / f"feature_bank_v30{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v30" / "feature_bank_v30.pth"


def ensure_current_bank(common_env: dict[str, str]) -> None:
    bank_suffix = common_env.get("BANK_SUFFIX", "")
    build_mode = env_str("BUILD_CURRENT_BANK", env_str("BUILD_V30_BANK", "auto")).lower()
    target_bank = bank_path_current(bank_suffix)
    should_build = build_mode == "1" or (build_mode == "auto" and not target_bank.exists())
    if should_build:
        print(f"[current] build feature bank: {target_bank}")
        run_cmd(BASE_DIR / "experiments" / "_feature_bank.py", common_env, status_label="current feature bank")
    else:
        print(f"[current] reuse feature bank: {target_bank}")


def git_show_to_file(commit: str, repo_rel_path: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "show", f"{commit}:{repo_rel_path}"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    dst.write_text(result.stdout, encoding="utf-8")


def progress_helper_block() -> str:
    return f"""
AB_PROGRESS_PREFIX = "{PROGRESS_PREFIX}"

def emit_ablation_progress(kind, **kwargs):
    if os.getenv("ABLATION_PROGRESS", "0").lower() not in {{"1", "true", "yes", "on"}}:
        return
    payload = {{"kind": kind, **kwargs}}
    print(AB_PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)

"""


def instrument_progress_script(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")
    if "emit_ablation_progress(" in text:
        return
    if "CONFIG = {" not in text:
        return
    text = text.replace("CONFIG = {", progress_helper_block() + "CONFIG = {", 1)
    replacements = [
        (
            'logger.info(f"共找到 {len(image_files)} 张图像")',
            'logger.info(f"共找到 {len(image_files)} 张图像")\n'
            '        emit_ablation_progress("category_start", category=category, current=int(start_idx), total=int(len(image_files)))',
        ),
        (
            'print(f"共找到 {len(image_files)} 张图像")',
            'print(f"共找到 {len(image_files)} 张图像")\n'
            '        emit_ablation_progress("category_start", category=category, current=int(start_idx), total=int(len(image_files)))',
        ),
        (
            'logger.info(f"从断点继续：start_idx={start_idx}")',
            'logger.info(f"从断点继续：start_idx={start_idx}")\n'
            '                emit_ablation_progress("resume", category=category, current=int(start_idx), total=int(len(image_files)))',
        ),
        (
            'print(f"从断点继续：start_idx={start_idx}")',
            'print(f"从断点继续：start_idx={start_idx}")\n'
            '                emit_ablation_progress("resume", category=category, current=int(start_idx), total=int(len(image_files)))',
        ),
        (
            '                img_file = image_files[i]',
            '                img_file = image_files[i]\n'
            '                emit_ablation_progress("image_progress", category=category, current=int(i + 1), total=int(len(image_files)), image=img_file)',
        ),
        (
            '        all_results[category] = category_results',
            '        emit_ablation_progress("category_done", category=category, current=int(last_completed), total=int(len(image_files)))\n'
            '        all_results[category] = category_results',
        ),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
    script_path.write_text(text, encoding="utf-8")


def materialize_legacy_version(version: str) -> dict[str, Path]:
    if version == "v7":
        commit = "bc76702"
        file_map = {
            "build": "experiments/01_build_feature_bank_v7.py",
            "infer": "experiments/02_inference_auto_v7.py",
            "eval": "experiments/03_evaluate_and_visualize_v7.py",
        }
    elif version == "v18":
        commit = "2dc7338"
        file_map = {
            "build": "experiments/01_build_feature_bank_v18.py",
            "infer": "experiments/02_inference_auto_v18.py",
            "eval": "experiments/03_evaluate_v18.py",
        }
    else:
        raise ValueError(f"Unsupported legacy version: {version}")

    materialized: dict[str, Path] = {}
    for key, repo_rel_path in file_map.items():
        dst = BASE_DIR / "experiments" / f"_materialized_{version}_{Path(repo_rel_path).name}"
        if not dst.exists() or env_flag("REFRESH_LEGACY_SCRIPTS", "0"):
            print(f"[legacy] materialize {version}: {repo_rel_path} @ {commit}")
            git_show_to_file(commit, repo_rel_path, dst)
        if key == "infer":
            instrument_progress_script(dst)
        materialized[key] = dst
    return materialized


def ensure_legacy_bank(version: str, scripts: dict[str, Path], common_env: dict[str, str]) -> None:
    build_mode = env_str(f"BUILD_{version.upper()}_BANK", "auto").lower()
    target_bank = BASE_DIR / "outputs" / f"feature_banks_{version}" / f"feature_bank_{version}.pth"
    should_build = build_mode == "1" or (build_mode == "auto" and not target_bank.exists())
    if should_build:
        print(f"[{version}] build feature bank: {target_bank}")
        run_cmd(scripts["build"], common_env, status_label=f"{version} feature bank")
    else:
        print(f"[{version}] reuse feature bank: {target_bank}")


def case_output_suffix(name: str) -> str:
    return f"_{name}"


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def metric_at_iou_05(results: dict) -> dict[str, float | int]:
    tp = 0
    fp = 0
    fn = 0
    for category, records in results.items():
        if category == "normal_auc":
            continue
        for record in records:
            if record.get("exclude_from_det_metrics"):
                continue
            pred_bboxes = record.get("pred_bboxes", []) or []
            gt_bboxes = record.get("gt_bboxes", []) or []
            matched_gt: set[int] = set()
            for pred in pred_bboxes:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gt_bboxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= 0.5:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn += len(gt_bboxes) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_pred": tp + fp,
    }


def get_image_score_strategies(record: dict) -> dict[str, float]:
    region_scores = record.get("anomaly_scores", []) or []
    region_max = max(region_scores) if region_scores else 0.0
    patch_max = float(record.get("max_patch_score", 0.0) or 0.0)
    patch_mean = float(record.get("mean_patch_score", 0.0) or 0.0)
    num_patches = int(record.get("num_patches", 0) or 0)
    num_anomaly_patches = int(record.get("num_anomaly_patches", 0) or 0)
    ratio = (float(num_anomaly_patches) / float(num_patches)) if num_patches > 0 else 0.0
    return {
        "region_max": region_max,
        "patch_mean": patch_mean,
        "patch_max": patch_max,
        "blend_region_patch": 0.5 * region_max + 0.5 * patch_max,
        "density_weighted_patch": patch_max * (1.0 + ratio),
    }


def safe_auc_from_report(report_obj: dict | None, preferred: str = "patch_mean") -> tuple[str, float | None]:
    if not report_obj:
        return preferred, None
    if "primary_image_score_strategy" in report_obj and "image_auc" in report_obj:
        strategy = str(report_obj.get("primary_image_score_strategy") or preferred)
        auc_value = report_obj.get("image_auc")
        return strategy, float(auc_value) if auc_value is not None else None
    return preferred, None


def patch_mean_gap(results: dict) -> tuple[float | None, float | None, float | None]:
    normal_scores: list[float] = []
    anomaly_scores: list[float] = []
    has_patch_info = False
    for category, records in results.items():
        for record in records:
            scores = get_image_score_strategies(record)
            if abs(scores["patch_mean"]) > 1e-12 or abs(scores["patch_max"]) > 1e-12:
                has_patch_info = True
            if category == "normal_auc":
                normal_scores.append(scores["patch_mean"])
            elif not record.get("exclude_from_det_metrics"):
                anomaly_scores.append(scores["patch_mean"])
    if not has_patch_info or not normal_scores or not anomaly_scores:
        return None, None, None
    normal_mean = float(np.mean(normal_scores))
    anomaly_mean = float(np.mean(anomaly_scores))
    return normal_mean, anomaly_mean, anomaly_mean - normal_mean


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_case(case: dict[str, object]) -> dict[str, object]:
    prediction_path = Path(str(case["prediction_path"]))
    report_path = Path(str(case["report_path"]))
    pred_obj = load_json_if_exists(prediction_path)
    if pred_obj is None:
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")
    results = pred_obj.get("results", {})
    metrics = metric_at_iou_05(results)
    report_obj = load_json_if_exists(report_path) if report_path.suffix.lower() == ".json" else None
    image_strategy, image_auc = safe_auc_from_report(report_obj, "patch_mean")
    patch_mean_normal, patch_mean_anomaly, patch_mean_gap_value = patch_mean_gap(results)
    pred_meta = pred_obj.get("meta", {})
    return {
        "name": case["name"],
        "family": case["family"],
        "description": case.get("description", ""),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_iou_05": metrics["f1"],
        "total_pred": metrics["total_pred"],
        "image_auc_strategy": image_strategy,
        "image_auc": image_auc,
        "patch_mean_normal": patch_mean_normal,
        "patch_mean_anomaly": patch_mean_anomaly,
        "patch_mean_gap": patch_mean_gap_value,
        "top_k_per_image": pred_meta.get("top_k_per_image"),
        "nms_iou_threshold": pred_meta.get("nms_iou_threshold"),
        "min_bbox_area": pred_meta.get("min_bbox_area"),
        "use_secondary_filter": pred_meta.get("use_secondary_filter"),
        "use_multi_scale": pred_meta.get("use_multi_scale"),
        "use_patchcore_topk_agg": pred_meta.get("use_patchcore_topk_agg"),
        "use_adaptive_beta": pred_meta.get("use_adaptive_beta"),
        "prediction_path": str(prediction_path),
        "report_path": str(report_path),
    }


def write_legacy_v18_report(prediction_path: Path, report_path: Path) -> None:
    pred_obj = load_json_if_exists(prediction_path)
    if pred_obj is None:
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")
    results = pred_obj.get("results", {})
    metrics = metric_at_iou_05(results)
    patch_mean_normal, patch_mean_anomaly, patch_mean_gap_value = patch_mean_gap(results)
    payload = {
        "version": "v18",
        "primary_image_score_strategy": "patch_mean",
        "image_auc": None,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "patch_mean_normal": patch_mean_normal,
        "patch_mean_anomaly": patch_mean_anomaly,
        "patch_mean_gap": patch_mean_gap_value,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "family",
        "description",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1_iou_05",
        "image_auc_strategy",
        "image_auc",
        "patch_mean_normal",
        "patch_mean_anomaly",
        "patch_mean_gap",
        "top_k_per_image",
        "nms_iou_threshold",
        "min_bbox_area",
        "use_secondary_filter",
        "use_multi_scale",
        "use_patchcore_topk_agg",
        "use_adaptive_beta",
        "prediction_path",
        "report_path",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Ablation Summary",
        "",
        "| Case | Family | F1@0.5 | P | R | TP | FP | FN | AUC | PatchMean Gap | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {family} | {f1} | {p} | {r} | {tp} | {fp} | {fn} | {auc} | {gap} | {desc} |".format(
                name=row["name"],
                family=row["family"],
                f1=fmt_float(row["f1_iou_05"]),
                p=fmt_float(row["precision"]),
                r=fmt_float(row["recall"]),
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
                auc=fmt_float(row["image_auc"]),
                gap=fmt_float(row["patch_mean_gap"]),
                desc=row["description"],
            )
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_ablation_results(manifest: list[dict[str, object]]) -> None:
    rows = [summarize_case(case) for case in manifest]
    write_csv(rows)
    write_markdown(rows)
    print(f"csv saved: {CSV_PATH}")
    print(f"markdown saved: {MD_PATH}")


CURRENT_CASES: list[dict[str, object]] = [
    {
        "name": "a0_final",
        "family": "current",
        "description": "Final current mainline: M1+M2+M3+M4+M5",
        "env": {},
    },
    {
        "name": "a1_wo_m5_secondary_filter",
        "family": "current",
        "description": "Disable secondary filter",
        "env": {
            "USE_SECONDARY_FILTER": "0",
        },
    },
    {
        "name": "a2_wo_m4_strict_filter",
        "family": "current",
        "description": "Relax top-k / nms / min-area back toward the V18 transition stage",
        "env": {
            "TOP_K_PER_IMAGE": "3",
            "NMS_IOU_THRESHOLD": "0.5",
            "MIN_BBOX_AREA": "3000",
            "BBOX_EXPAND_PIXELS": "15",
        },
    },
    {
        "name": "a5_wo_m1_beta_calibration",
        "family": "current",
        "description": "Disable validation-driven beta calibration",
        "env": {
            "USE_ADAPTIVE_BETA": "0",
        },
    },
    {
        "name": "c1_patchcore_auc",
        "family": "current",
        "description": "Enable PatchCore-style image-level aggregation control",
        "env": {
            "USE_PATCHCORE_TOPK_AGG": "1",
            "PATCHCORE_TOPK": env_str("PATCHCORE_TOPK", "3"),
            "PATCHCORE_REWEIGHT_LAMBDA": env_str("PATCHCORE_REWEIGHT_LAMBDA", "0.35"),
        },
    },
    {
        "name": "c2_single_scale_auc_control",
        "family": "current",
        "description": "Disable multiscale fusion to measure its AUC-side contribution",
        "env": {
            "USE_MULTI_SCALE": "0",
        },
    },
]


LEGACY_CASES: list[dict[str, object]] = [
    {
        "name": "s0_v7_baseline",
        "family": "v7",
        "description": "Original candidate-driven baseline",
    },
    {
        "name": "s1_v18_transition",
        "family": "v18",
        "description": "Full-image score map + connected components transition stage",
    },
]


ALL_CASES = CURRENT_CASES + LEGACY_CASES


def selected_cases() -> list[dict[str, object]]:
    chosen = env_str("WORKER_CASES", env_str("ABLATION_CASES", ""))
    if not chosen:
        return list(ALL_CASES)
    name_set = {part.strip() for part in chosen.split(",") if part.strip()}
    cases = [case for case in ALL_CASES if str(case["name"]) in name_set]
    missing = sorted(name_set.difference({str(case["name"]) for case in cases}))
    if missing:
        raise ValueError(f"Unknown ABLATION_CASES: {missing}")
    return cases


def selected_case_names() -> list[str]:
    return [str(case["name"]) for case in selected_cases()]


def build_case_lookup() -> dict[str, dict[str, object]]:
    return {str(case["name"]): case for case in ALL_CASES}


def ordered_case_names() -> list[str]:
    return [str(case["name"]) for case in ALL_CASES]


def split_case_names(case_names: list[str], num_groups: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(max(1, num_groups))]
    for idx, name in enumerate(case_names):
        groups[idx % len(groups)].append(name)
    return [group for group in groups if group]


def load_manifest_cases() -> dict[str, dict[str, object]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cases = payload.get("cases", []) or []
    return {str(case.get("name")): case for case in cases if case.get("name")}


def report_exists(case: dict[str, object]) -> bool:
    report_path = Path(str(case["report_path"]))
    prediction_path = Path(str(case["prediction_path"]))
    return report_path.exists() and prediction_path.exists()


def manifest_entry_for_case(case_name: str) -> dict[str, object]:
    case = build_case_lookup()[case_name]
    family = str(case["family"])
    description = str(case.get("description", ""))
    if family == "current":
        suffix = case_output_suffix(case_name)
        return {
            "name": case_name,
            "family": family,
            "description": description,
            "prediction_path": str(BASE_DIR / "outputs" / "predictions_v30" / f"auto_predictions_v30{suffix}.json"),
            "report_path": str(BASE_DIR / "outputs" / "predictions_v30" / f"evaluation_report_v30{suffix}.json"),
        }
    if family == "v7":
        return {
            "name": case_name,
            "family": family,
            "description": description,
            "prediction_path": str(BASE_DIR / "outputs" / "predictions_v7" / "auto_predictions_v7.json"),
            "report_path": str(BASE_DIR / "outputs" / "visualizations_v7" / "03_evaluate_and_visualize_v7_report.md"),
        }
    if family == "v18":
        return {
            "name": case_name,
            "family": family,
            "description": description,
            "prediction_path": str(BASE_DIR / "outputs" / "predictions_v18" / "auto_predictions_v18.json"),
            "report_path": str(BASE_DIR / "outputs" / "predictions_v18" / "evaluation_report_v18.json"),
        }
    raise ValueError(f"Unsupported case family: {family}")


def manifest_entries_for_case_names(case_names: list[str]) -> list[dict[str, object]]:
    return [manifest_entry_for_case(name) for name in case_names]


def manifest_stub_for_case(case_name: str) -> dict[str, object]:
    entry = manifest_entry_for_case(case_name)
    entry.update(
        {
            "status": "pending",
            "stage": "pending",
            "worker": "",
            "started_at": None,
            "updated_at": now_iso(),
            "finished_at": None,
            "last_error": "",
            "note": "",
            "case_index": None,
            "case_total": None,
        }
    )
    return entry


def ordered_manifest_cases(case_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    names = [name for name in ordered_case_names() if name in case_map]
    extras = [name for name in case_map.keys() if name not in names]
    names.extend(sorted(extras))
    return [case_map[name] for name in names]


def write_case_status_file(case_data: dict[str, object]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATUS_DIR / f"{case_data['name']}.json"
    status_path.write_text(json.dumps(case_data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_manifest_case(case_name: str, **updates: object) -> dict[str, object]:
    with manifest_lock():
        case_map = load_manifest_cases()
        current = manifest_stub_for_case(case_name)
        current.update(case_map.get(case_name, {}))
        current.update(updates)
        current["updated_at"] = now_iso()
        if current.get("status") == "running" and not current.get("started_at"):
            current["started_at"] = current["updated_at"]
        if current.get("status") in {"completed", "failed", "skipped"}:
            current["finished_at"] = current["updated_at"]
        case_map[case_name] = current
        write_manifest(ordered_manifest_cases(case_map))
    write_case_status_file(current)
    return current


def ensure_manifest_cases(case_names: list[str]) -> None:
    with manifest_lock():
        case_map = load_manifest_cases()
        changed = False
        for case_name in case_names:
            if case_name not in case_map:
                case_map[case_name] = manifest_stub_for_case(case_name)
                changed = True
        if changed:
            write_manifest(ordered_manifest_cases(case_map))
    for case_name in case_names:
        case_data = load_manifest_cases().get(case_name)
        if case_data:
            write_case_status_file(case_data)


def format_case_prefix(worker_label: str, case_name: str, case_index: int, case_total: int, stage: str) -> str:
    return f"[{worker_label}][case {case_index}/{case_total}][{case_name}][{stage}]"


def print_case_progress(worker_label: str, case_name: str, case_index: int, case_total: int, stage: str, message: str) -> None:
    print(f"{format_case_prefix(worker_label, case_name, case_index, case_total, stage)} {message}")


def manifest_progress_snapshot(case_names: list[str]) -> str:
    case_map = load_manifest_cases()
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
    running_items: list[str] = []
    for case_name in case_names:
        case_data = case_map.get(case_name, manifest_stub_for_case(case_name))
        status = str(case_data.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
        if status == "running":
            worker = str(case_data.get("worker") or "-")
            stage = str(case_data.get("stage") or "-")
            running_items.append(f"{worker}:{case_name}:{stage}")
    summary = (
        f"[progress] total={len(case_names)} completed={counts.get('completed', 0)} "
        f"running={counts.get('running', 0)} pending={counts.get('pending', 0)} "
        f"failed={counts.get('failed', 0)} skipped={counts.get('skipped', 0)}"
    )
    if running_items:
        summary += " | active=" + ", ".join(running_items)
    return summary


def filter_pending_case_names(case_names: list[str]) -> list[str]:
    pending: list[str] = []
    for name in case_names:
        manifest_case = manifest_entry_for_case(name)
        if manifest_case and report_exists(manifest_case):
            print(f"[skip] completed case detected: {name}")
            update_manifest_case(name, status="completed", stage="done", note="existing outputs detected")
            continue
        pending.append(name)
    return pending


def build_common_env() -> dict[str, str]:
    common_env: dict[str, str] = {}
    bank_suffix = env_str("BANK_SUFFIX", "")
    if bank_suffix:
        common_env["BANK_SUFFIX"] = bank_suffix
    max_images = env_str("MAX_IMAGES_PER_CATEGORY", "")
    if max_images:
        common_env["MAX_IMAGES_PER_CATEGORY"] = max_images
    return common_env


def run_current_case(
    case: dict[str, object],
    common_env: dict[str, str],
    manifest: list[dict[str, object]],
    *,
    worker_label: str,
    case_index: int,
    case_total: int,
) -> None:
    case_name = str(case["name"])
    output_suffix = case_output_suffix(str(case["name"]))
    case_env = {
        **common_env,
        "OUTPUT_SUFFIX": output_suffix,
        **dict(case.get("env", {})),
    }
    prediction_path = BASE_DIR / "outputs" / "predictions_v30" / f"auto_predictions_v30{output_suffix}.json"
    report_path = BASE_DIR / "outputs" / "predictions_v30" / f"evaluation_report_v30{output_suffix}.json"
    print(f"\n=== CASE {case_name} (current) ===")
    print(case.get("description", ""))
    update_manifest_case(
        case_name,
        status="running",
        stage="inference",
        worker=worker_label,
        note="current pipeline started",
        case_index=case_index,
        case_total=case_total,
        prediction_path=str(prediction_path),
        report_path=str(report_path),
    )
    print_case_progress(worker_label, case_name, case_index, case_total, "inference", "start; child script should show image progress such as 1/30")
    run_cmd(
        BASE_DIR / "experiments" / "_inference_auto.py",
        case_env,
        status_label=f"{case_name} inference",
    )
    update_manifest_case(case_name, status="running", stage="evaluate", worker=worker_label, note="inference done")
    print_case_progress(worker_label, case_name, case_index, case_total, "evaluate", "start")
    run_cmd(
        BASE_DIR / "experiments" / "_evaluate.py",
        case_env,
        status_label=f"{case_name} evaluate",
    )
    update_manifest_case(
        case_name,
        status="completed",
        stage="done",
        worker=worker_label,
        note="current pipeline completed",
        prediction_path=str(prediction_path),
        report_path=str(report_path),
    )
    print_case_progress(worker_label, case_name, case_index, case_total, "done", f"prediction={prediction_path.name} report={report_path.name}")
    manifest.append(
        {
            "name": case["name"],
            "family": "current",
            "description": case.get("description", ""),
            "prediction_path": str(prediction_path),
            "report_path": str(report_path),
        }
    )


def run_legacy_case(
    case: dict[str, object],
    common_env: dict[str, str],
    manifest: list[dict[str, object]],
    *,
    worker_label: str,
    case_index: int,
    case_total: int,
) -> None:
    case_name = str(case["name"])
    version = str(case["family"])
    scripts = materialize_legacy_version(version)
    ensure_legacy_bank(version, scripts, common_env)
    print(f"\n=== CASE {case_name} ({version}) ===")
    print(case.get("description", ""))
    update_manifest_case(
        case_name,
        status="running",
        stage="inference",
        worker=worker_label,
        note=f"{version} pipeline started",
        case_index=case_index,
        case_total=case_total,
    )
    print_case_progress(worker_label, case_name, case_index, case_total, "inference", "start; child script should show image progress such as 1/30")
    run_cmd(scripts["infer"], common_env, status_label=f"{case_name} inference")

    if version == "v7":
        prediction_path = BASE_DIR / "outputs" / "predictions_v7" / "auto_predictions_v7.json"
        report_path = BASE_DIR / "outputs" / "visualizations_v7" / "03_evaluate_and_visualize_v7_report.md"
        update_manifest_case(
            case_name,
            status="running",
            stage="evaluate",
            worker=worker_label,
            note="legacy v7 inference done",
            prediction_path=str(prediction_path),
            report_path=str(report_path),
        )
        print_case_progress(worker_label, case_name, case_index, case_total, "evaluate", "start")
        run_cmd(scripts["eval"], common_env, status_label=f"{case_name} evaluate")
    elif version == "v18":
        prediction_path = BASE_DIR / "outputs" / "predictions_v18" / "auto_predictions_v18.json"
        report_path = BASE_DIR / "outputs" / "predictions_v18" / "evaluation_report_v18.json"
        update_manifest_case(
            case_name,
            status="running",
            stage="evaluate",
            worker=worker_label,
            note="legacy v18 inference done",
            prediction_path=str(prediction_path),
            report_path=str(report_path),
        )
        print_case_progress(worker_label, case_name, case_index, case_total, "evaluate", "start")
        write_legacy_v18_report(prediction_path, report_path)
    else:
        raise ValueError(version)

    update_manifest_case(
        case_name,
        status="completed",
        stage="done",
        worker=worker_label,
        note=f"{version} pipeline completed",
        prediction_path=str(prediction_path),
        report_path=str(report_path),
    )
    print_case_progress(worker_label, case_name, case_index, case_total, "done", f"prediction={prediction_path.name} report={report_path.name}")
    manifest.append(
        {
            "name": case["name"],
            "family": version,
            "description": case.get("description", ""),
            "prediction_path": str(prediction_path),
            "report_path": str(report_path),
        }
    )


def run_selected_cases(common_env: dict[str, str]) -> list[dict[str, object]]:
    cases = selected_cases()
    worker_label = env_str("WORKER_LABEL", "main")
    print(f"BASE_DIR={BASE_DIR}")
    print(f"TOTAL_CASES={len(cases)}")
    print(f"WORKER={worker_label}")
    print("CASES=", ", ".join(str(case["name"]) for case in cases))
    ensure_manifest_cases([str(case["name"]) for case in cases])

    if any(str(case["family"]) == "current" for case in cases):
        ensure_current_bank(common_env)

    manifest: list[dict[str, object]] = []
    total_cases = len(cases)
    for idx, case in enumerate(cases, 1):
        family = str(case["family"])
        case_name = str(case["name"])
        print_case_progress(worker_label, case_name, idx, total_cases, "prepare", str(case.get("description", "")))
        try:
            if family == "current":
                run_current_case(
                    case,
                    common_env,
                    manifest,
                    worker_label=worker_label,
                    case_index=idx,
                    case_total=total_cases,
                )
            else:
                run_legacy_case(
                    case,
                    common_env,
                    manifest,
                    worker_label=worker_label,
                    case_index=idx,
                    case_total=total_cases,
                )
        except Exception as exc:
            update_manifest_case(
                case_name,
                status="failed",
                stage="failed",
                worker=worker_label,
                note=f"{family} pipeline failed",
                last_error=str(exc),
                case_index=idx,
                case_total=total_cases,
            )
            print_case_progress(worker_label, case_name, idx, total_cases, "failed", str(exc))
            raise
    return manifest


def merge_manifest_cases(new_cases: list[dict[str, object]]) -> list[dict[str, object]]:
    merged = load_manifest_cases()
    for case in new_cases:
        merged[str(case["name"])] = case
    names = [name for name in ordered_case_names() if name in merged]
    extras = [name for name in merged.keys() if name not in names]
    names.extend(sorted(extras))
    return [merged[name] for name in names]


def write_manifest(cases: list[dict[str, object]]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "base_dir": str(BASE_DIR),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_worker_entry() -> int:
    common_env = build_common_env()
    new_cases = run_selected_cases(common_env)
    print(f"\nworker completed cases: {', '.join(case['name'] for case in new_cases)}")
    return 0


def run_parallel_supervisor() -> int:
    selected_names = selected_case_names()
    if not selected_names:
        print("No cases selected.")
        return 0
    ensure_manifest_cases(selected_names)

    skip_completed = env_flag("SKIP_COMPLETED_CASES", "1")
    if skip_completed:
        selected_names = filter_pending_case_names(selected_names)
    if not selected_names:
        print("All selected cases already completed.")
        merged_cases = manifest_entries_for_case_names(selected_case_names())
        write_manifest(merged_cases)
        if merged_cases:
            collect_ablation_results(merged_cases)
        return 0

    common_env = {}
    common_env = build_common_env()

    case_lookup = build_case_lookup()
    selected_families = {str(case_lookup[name]["family"]) for name in selected_names}
    if "current" in selected_families:
        ensure_current_bank(common_env)
    if "v7" in selected_families:
        scripts_v7 = materialize_legacy_version("v7")
        ensure_legacy_bank("v7", scripts_v7, common_env)
    if "v18" in selected_families:
        scripts_v18 = materialize_legacy_version("v18")
        ensure_legacy_bank("v18", scripts_v18, common_env)

    num_workers = min(env_int("PARALLEL_WORKERS", 2), len(selected_names))
    restart_delay_sec = max(1, env_int("WORKER_RESTART_DELAY_SEC", 5))
    max_restarts = env_int("WORKER_MAX_RESTARTS", 20)
    groups = split_case_names(selected_names, num_workers)

    print(f"[parallel] workers={len(groups)}")
    for idx, group in enumerate(groups, 1):
        print(f"[parallel] worker_{idx}: {', '.join(group)}")

    script_path = Path(__file__).resolve()
    restart_counts = [0 for _ in groups]
    completed = [False for _ in groups]

    procs: list[subprocess.Popen[str] | None] = []
    for idx, group in enumerate(groups, 1):
        worker_env = {
            **common_env,
            "WORKER_MODE": "1",
            "WORKER_CASES": ",".join(group),
            "BUILD_CURRENT_BANK": "0",
            "BUILD_V7_BANK": "0",
            "BUILD_V18_BANK": "0",
            "SKIP_COMPLETED_CASES": "1",
            "WORKER_LABEL": f"worker_{idx}",
        }
        procs.append(run_python_module(script_path, worker_env))

    last_progress_log = 0.0
    last_progress_snapshot = ""
    while not all(completed):
        time.sleep(5)
        now = time.time()
        if now - last_progress_log >= 15:
            snapshot = manifest_progress_snapshot(selected_names)
            if snapshot != last_progress_snapshot or now - last_progress_log >= 30:
                print(snapshot)
                last_progress_snapshot = snapshot
            last_progress_log = now
        for idx, proc in enumerate(procs):
            if completed[idx] or proc is None:
                continue
            return_code = proc.poll()
            if return_code is None:
                continue
            if return_code == 0:
                completed[idx] = True
                print(f"[parallel] worker_{idx + 1} completed")
                continue

            restart_counts[idx] += 1
            print(f"[parallel] worker_{idx + 1} exited with code {return_code}, restart {restart_counts[idx]}/{max_restarts}")
            if restart_counts[idx] > max_restarts:
                raise RuntimeError(f"worker_{idx + 1} exceeded max restarts")

            time.sleep(restart_delay_sec)
            group = groups[idx]
            worker_env = {
                **common_env,
                "WORKER_MODE": "1",
                "WORKER_CASES": ",".join(group),
                "BUILD_CURRENT_BANK": "0",
                "BUILD_V7_BANK": "0",
                "BUILD_V18_BANK": "0",
                "SKIP_COMPLETED_CASES": "1",
                "WORKER_LABEL": f"worker_{idx + 1}",
            }
            procs[idx] = run_python_module(script_path, worker_env)

    merged_case_map = load_manifest_cases()
    merged_cases = [merged_case_map.get(name, manifest_stub_for_case(name)) for name in selected_case_names()]
    write_manifest(merged_cases)
    if merged_cases:
        collect_ablation_results([case for case in merged_cases if report_exists(case)])
    return 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if env_flag("WORKER_MODE", "0"):
        return run_worker_entry()
    if not env_flag("FORCE_SEQUENTIAL", "0") and env_flag("RUN_PARALLEL", "1"):
        return run_parallel_supervisor()

    ensure_manifest_cases(selected_case_names())
    manifest = run_selected_cases(build_common_env())
    merged_case_map = load_manifest_cases()
    merged_manifest = [merged_case_map.get(name, manifest_stub_for_case(name)) for name in selected_case_names()]
    write_manifest(merged_manifest)
    print(f"\nmanifest saved: {MANIFEST_PATH}")
    collect_ablation_results([case for case in merged_manifest if report_exists(case)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
