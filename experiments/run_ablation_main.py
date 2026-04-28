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
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUTPUT_DIR = BASE_DIR / "outputs" / "ablation_suite"
MANIFEST_PATH = OUTPUT_DIR / "case_manifest.json"
CSV_PATH = OUTPUT_DIR / "ablation_summary.csv"
MD_PATH = OUTPUT_DIR / "ablation_summary.md"


def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


def env_flag(name: str, default: str = "0") -> bool:
    return env_str(name, default).lower() in {"1", "true", "yes", "on"}


def run_cmd(script_path: Path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)


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
        run_cmd(BASE_DIR / "experiments" / "_feature_bank.py", common_env)
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
        materialized[key] = dst
    return materialized


def ensure_legacy_bank(version: str, scripts: dict[str, Path], common_env: dict[str, str]) -> None:
    build_mode = env_str(f"BUILD_{version.upper()}_BANK", "auto").lower()
    target_bank = BASE_DIR / "outputs" / f"feature_banks_{version}" / f"feature_bank_{version}.pth"
    should_build = build_mode == "1" or (build_mode == "auto" and not target_bank.exists())
    if should_build:
        print(f"[{version}] build feature bank: {target_bank}")
        run_cmd(scripts["build"], common_env)
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
    chosen = env_str("ABLATION_CASES", "")
    if not chosen:
        return list(ALL_CASES)
    name_set = {part.strip() for part in chosen.split(",") if part.strip()}
    cases = [case for case in ALL_CASES if str(case["name"]) in name_set]
    missing = sorted(name_set.difference({str(case["name"]) for case in cases}))
    if missing:
        raise ValueError(f"Unknown ABLATION_CASES: {missing}")
    return cases


def run_current_case(case: dict[str, object], common_env: dict[str, str], manifest: list[dict[str, object]]) -> None:
    output_suffix = case_output_suffix(str(case["name"]))
    case_env = {
        **common_env,
        "OUTPUT_SUFFIX": output_suffix,
        **dict(case.get("env", {})),
    }
    print(f"\n=== CASE {case['name']} (current) ===")
    print(case.get("description", ""))
    run_cmd(BASE_DIR / "experiments" / "_inference_auto.py", case_env)
    run_cmd(BASE_DIR / "experiments" / "_evaluate.py", case_env)
    manifest.append(
        {
            "name": case["name"],
            "family": "current",
            "description": case.get("description", ""),
            "prediction_path": str(BASE_DIR / "outputs" / "predictions_v30" / f"auto_predictions_v30{output_suffix}.json"),
            "report_path": str(BASE_DIR / "outputs" / "predictions_v30" / f"evaluation_report_v30{output_suffix}.json"),
        }
    )


def run_legacy_case(case: dict[str, object], common_env: dict[str, str], manifest: list[dict[str, object]]) -> None:
    version = str(case["family"])
    scripts = materialize_legacy_version(version)
    ensure_legacy_bank(version, scripts, common_env)
    print(f"\n=== CASE {case['name']} ({version}) ===")
    print(case.get("description", ""))
    run_cmd(scripts["infer"], common_env)
    run_cmd(scripts["eval"], common_env)

    if version == "v7":
        prediction_path = BASE_DIR / "outputs" / "predictions_v7" / "auto_predictions_v7.json"
        report_path = BASE_DIR / "outputs" / "visualizations_v7" / "03_evaluate_and_visualize_v7_report.md"
    elif version == "v18":
        prediction_path = BASE_DIR / "outputs" / "predictions_v18" / "auto_predictions_v18.json"
        report_path = BASE_DIR / "outputs" / "predictions_v18" / "evaluation_report_v18.json"
    else:
        raise ValueError(version)

    manifest.append(
        {
            "name": case["name"],
            "family": version,
            "description": case.get("description", ""),
            "prediction_path": str(prediction_path),
            "report_path": str(report_path),
        }
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common_env = {}
    bank_suffix = env_str("BANK_SUFFIX", "")
    if bank_suffix:
        common_env["BANK_SUFFIX"] = bank_suffix
    max_images = env_str("MAX_IMAGES_PER_CATEGORY", "")
    if max_images:
        common_env["MAX_IMAGES_PER_CATEGORY"] = max_images

    cases = selected_cases()
    print(f"BASE_DIR={BASE_DIR}")
    print(f"TOTAL_CASES={len(cases)}")
    print("CASES=", ", ".join(str(case["name"]) for case in cases))

    if any(str(case["family"]) == "current" for case in cases):
        ensure_current_bank(common_env)

    manifest: list[dict[str, object]] = []
    for case in cases:
        family = str(case["family"])
        if family == "current":
            run_current_case(case, common_env, manifest)
        else:
            run_legacy_case(case, common_env, manifest)

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "base_dir": str(BASE_DIR),
                "cases": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nmanifest saved: {MANIFEST_PATH}")
    collect_ablation_results(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
