"""
V26 parallel quick sweep runner.

用途：
- 自动拆成两路并行运行
- 每个 case 使用独立 output suffix，bank 侧 case 使用独立 bank suffix
- 父进程定时读取 checkpoint，汇总显示当前 case 进度
- 子进程异常退出后自动重启，并从已有 checkpoint 继续
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v26_quick_sweep.py"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints_v26"
PREDICTION_DIR = BASE_DIR / "outputs" / "predictions_v26"
LOG_ROOT_DIR = BASE_DIR / "outputs" / "parallel_logs_v26"


def env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


def build_common_env() -> dict[str, str]:
    bank_root_suffix = env_str("BANK_SUFFIX", "")
    return {
        "BANK_SUFFIX": bank_root_suffix,
        "BANK_ROOT_SUFFIX": bank_root_suffix,
        "MAX_IMAGES_PER_CATEGORY": env_str("MAX_IMAGES_PER_CATEGORY", "30"),
        "SECONDARY_FILTER_MIN_MEAN_PATCH": env_str("SECONDARY_FILTER_MIN_MEAN_PATCH", "0.275"),
        "SECONDARY_FILTER_BOX_SCORE_MIN": env_str("SECONDARY_FILTER_BOX_SCORE_MIN", "0.0"),
        "SOFTPATCH_KEEP_RATIO": env_str("SOFTPATCH_KEEP_RATIO", "0.85"),
        "SOFTPATCH_KNN_K": env_str("SOFTPATCH_KNN_K", "5"),
        "SOFTPATCH_SCORE_POWER": env_str("SOFTPATCH_SCORE_POWER", "1.0"),
        "PATCHCORE_TOPK": env_str("PATCHCORE_TOPK", "3"),
        "PATCHCORE_REWEIGHT_LAMBDA": env_str("PATCHCORE_REWEIGHT_LAMBDA", "0.35"),
        "THRESHOLD_NORMAL_STD_MULT": env_str("THRESHOLD_NORMAL_STD_MULT", "2.5"),
        "THRESHOLD_IMAGE_STD_MULT": env_str("THRESHOLD_IMAGE_STD_MULT", "1.0"),
        "THRESHOLD_MIN_FLOOR_DELTA": env_str("THRESHOLD_MIN_FLOOR_DELTA", "0.0"),
    }


def build_case(name: str, bank_tag: str, **env_overrides: str) -> dict[str, object]:
    return {
        "name": name,
        "bank_tag": bank_tag,
        "env": env_overrides,
    }


ALL_CASES = [
    build_case("base", "base"),
    build_case("softpatch", "softpatch", USE_SOFTPATCH_BANK_FILTER="1"),
    build_case("patchcore", "base", USE_PATCHCORE_TOPK_AGG="1"),
    build_case("threshold", "base", USE_THRESHOLD_LEARNER="1"),
    build_case(
        "softpatch_patchcore",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_PATCHCORE_TOPK_AGG="1",
    ),
    build_case(
        "softpatch_threshold",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_THRESHOLD_LEARNER="1",
    ),
    build_case(
        "patchcore_threshold",
        "base",
        USE_PATCHCORE_TOPK_AGG="1",
        USE_THRESHOLD_LEARNER="1",
    ),
    build_case(
        "all3",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_PATCHCORE_TOPK_AGG="1",
        USE_THRESHOLD_LEARNER="1",
    ),
]

CASE_BY_NAME = {str(case["name"]): case for case in ALL_CASES}
CATEGORY_DIRS = {
    "cavities": BASE_DIR / "data" / "GPR_data" / "augmented_cavities",
    "utilities": BASE_DIR / "data" / "GPR_data" / "augmented_utilities",
    "normal_auc": BASE_DIR / "data" / "GPR_data" / "augmented_intact",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def case_output_suffix(case_name: str) -> str:
    return f"_{case_name}"


def case_bank_suffix(root_suffix: str, bank_tag: str) -> str:
    suffix = root_suffix or ""
    if bank_tag == "base":
        return suffix
    return f"{suffix}_{bank_tag}" if suffix else f"_{bank_tag}"


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v26{bank_suffix}" / f"feature_bank_v26{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v26" / "feature_bank_v26.pth"


def resolve_build_bank(mode: str, bank_suffix: str) -> bool:
    lower = mode.lower()
    if lower == "1":
        return True
    if lower == "0":
        return False
    if lower == "auto":
        return not bank_file_path(bank_suffix).exists()
    raise ValueError(f"Unsupported BUILD_BANK={mode!r}, expected auto/0/1")


def count_images(data_dir: Path, max_images_per_category: int | None) -> int:
    if not data_dir.exists():
        return 0
    files = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if max_images_per_category is not None:
        return min(len(files), max_images_per_category)
    return len(files)


def build_category_totals(common_env: dict[str, str]) -> dict[str, int]:
    raw = env_str("MAX_IMAGES_PER_CATEGORY", common_env["MAX_IMAGES_PER_CATEGORY"])
    limit = int(raw) if raw else None
    return {
        category: count_images(path, limit)
        for category, path in CATEGORY_DIRS.items()
    }


def checkpoint_path(case_name: str, category: str) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_auto_{category}{case_output_suffix(case_name)}.json"


def prediction_path(case_name: str) -> Path:
    return PREDICTION_DIR / f"auto_predictions_v26{case_output_suffix(case_name)}.json"


def report_path(case_name: str) -> Path:
    return PREDICTION_DIR / f"evaluation_report_v26{case_output_suffix(case_name)}.json"


def case_is_done(case_name: str) -> bool:
    return report_path(case_name).exists()


def read_json(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_single_case(case_name: str, category_totals: dict[str, int]) -> tuple[str, bool]:
    category_chunks: list[str] = []
    completed_categories = 0
    seen_checkpoint = False

    for category in ["cavities", "utilities", "normal_auc"]:
        total = int(category_totals.get(category, 0))
        info = read_json(checkpoint_path(case_name, category))
        processed = 0
        completed = False
        if info:
            seen_checkpoint = True
            processed = int(info.get("processed_count", 0) or 0)
            completed = bool(info.get("completed", False))
            if completed and total > 0:
                processed = total
        if completed:
            completed_categories += 1
        short_name = {"cavities": "cav", "utilities": "util", "normal_auc": "normal"}[category]
        category_chunks.append(f"{short_name} {processed}/{total}")

    if report_path(case_name).exists():
        stage = "done"
    elif prediction_path(case_name).exists() and completed_categories == 3:
        stage = "evaluating"
    elif seen_checkpoint:
        stage = "running"
    else:
        stage = "pending"

    return f"{case_name}:{stage} [{' | '.join(category_chunks)}]", stage == "done"


def summarize_runner(runner: dict, category_totals: dict[str, int]) -> str:
    case_summaries: list[str] = []
    done_cases = 0
    current_case = "unknown"

    for idx in runner["case_indices"]:
        case_name = str(ALL_CASES[idx]["name"])
        summary, finished = summarize_single_case(case_name, category_totals)
        case_summaries.append(summary)
        if finished:
            done_cases += 1
        elif current_case == "unknown":
            current_case = case_name

    if current_case == "unknown" and case_summaries:
        current_case = str(ALL_CASES[runner["case_indices"][-1]]["name"])

    proc = runner["proc"]
    if all(case_is_done(str(ALL_CASES[idx]["name"])) for idx in runner["case_indices"]):
        stage = "done"
    elif proc is not None and proc.poll() is None:
        stage = f"running:{current_case}"
    else:
        stage = "waiting_restart"

    return (
        f"[{runner['title']}] cases {done_cases}/{len(runner['case_indices'])} | "
        f"{stage} | restart={runner['restart_count']} | "
        f"log={runner['log_path'].name} | " + " || ".join(case_summaries)
    )


def open_log_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)


def spawn_runner(case_indices: list[int], build_bank: str, title: str,
                 common_env: dict[str, str], log_dir: Path, restart_count: int) -> tuple[subprocess.Popen, Path]:
    env = os.environ.copy()
    env.update(common_env)
    env["BUILD_BANK"] = build_bank
    env["V26_CASE_INDICES"] = ",".join(str(i) for i in case_indices)
    env["V26_SWEEP_TITLE"] = title

    first_case_name = str(ALL_CASES[case_indices[0]]["name"])
    log_path = log_dir / f"{title.lower().replace(' ', '_')}_{first_case_name}.log"
    log_file = open_log_file(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n===== launch #{restart_count + 1} @ {timestamp} =====\n")

    cmd = [PYTHON, str(SCRIPT_07)]
    print(">>>", " ".join(cmd), f"[{title}] restart={restart_count}")
    print(f"    log -> {log_path}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_codex_log_file", log_file)
    return proc, log_path


def close_proc_log(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    log_file = getattr(proc, "_codex_log_file", None)
    if log_file is not None:
        log_file.close()


def read_log_tail(log_path: Path, max_lines: int = 20) -> list[str]:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return []
    return [line.rstrip("\r\n") for line in lines[-max_lines:]]


def describe_return_code(returncode: int | None) -> str:
    if returncode is None:
        return "running"
    if returncode == 0:
        return "ok"
    if returncode < 0:
        sig = -returncode
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = f"SIG{sig}"
        if sig == 9:
            return f"killed_by_signal({sig_name}) possible_oom"
        return f"killed_by_signal({sig_name})"
    if returncode == 137:
        return "exit=137 possible_oom"
    if returncode == 143:
        return "exit=143 terminated"
    return f"exit={returncode}"


def report_runner_failure(runner: dict) -> str:
    proc = runner["proc"]
    status = describe_return_code(proc.poll())
    print("\n=== PARALLEL FAILURE DETECTED ===")
    print(f"runner={runner['title']}")
    print(f"status={status}")
    print(f"log={runner['log_path']}")
    tail_lines = read_log_tail(runner["log_path"], max_lines=25)
    if tail_lines:
        print("--- log tail ---")
        for line in tail_lines:
            print(line)
    return status


def restart_runner(runner: dict, common_env: dict[str, str], log_dir: Path) -> None:
    close_proc_log(runner["proc"])
    runner["restart_count"] += 1
    proc, log_path = spawn_runner(
        runner["case_indices"], "0", runner["title"], common_env, log_dir, runner["restart_count"]
    )
    runner["proc"] = proc
    runner["log_path"] = log_path
    runner["last_failure"] = ""


def monitor_parallel_runs(runners: list[dict], category_totals: dict[str, int],
                          common_env: dict[str, str], log_dir: Path,
                          poll_interval: float = 1.0) -> dict[str, str]:
    last_snapshot = ""
    failures: dict[str, str] = {}
    while True:
        lines = [summarize_runner(runner, category_totals) for runner in runners]
        snapshot = "\n".join(lines)
        if snapshot != last_snapshot:
            print("\n=== PARALLEL PROGRESS ===")
            for line in lines:
                print(line)
            last_snapshot = snapshot

        for runner in runners:
            proc = runner["proc"]
            rc = proc.poll()
            if rc is None or rc == 0:
                continue
            failure_key = runner["title"]
            if failure_key in failures:
                continue

            status = report_runner_failure(runner)
            remaining_cases = [
                str(ALL_CASES[idx]["name"])
                for idx in runner["case_indices"]
                if not case_is_done(str(ALL_CASES[idx]["name"]))
            ]
            if runner["restart_count"] < runner["max_restarts"] and remaining_cases:
                print(
                    f"[monitor] restarting {runner['title']} "
                    f"({runner['restart_count'] + 1}/{runner['max_restarts']}) from checkpoint"
                )
                restart_runner(runner, common_env, log_dir)
                last_snapshot = ""
                continue

            failures[failure_key] = status

        if all(
            all(case_is_done(str(ALL_CASES[idx]["name"])) for idx in runner["case_indices"]) or
            (runner["proc"].poll() is not None and runner["title"] in failures) or
            (runner["proc"].poll() == 0)
            for runner in runners
        ):
            break
        time.sleep(poll_interval)
    return failures


def split_case_indices(total_cases: int) -> tuple[list[int], list[int]]:
    midpoint = (total_cases + 1) // 2
    return list(range(0, midpoint)), list(range(midpoint, total_cases))


def main() -> int:
    common_env = build_common_env()
    bank_root_suffix = common_env["BANK_ROOT_SUFFIX"]
    category_totals = build_category_totals(common_env)
    log_dir = LOG_ROOT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    max_restarts = int(env_str("MAX_RESTARTS_PER_RUNNER", "2"))

    first_half, second_half = split_case_indices(len(ALL_CASES))
    pending_first_half = [idx for idx in first_half if not case_is_done(str(ALL_CASES[idx]["name"]))]
    pending_second_half = [idx for idx in second_half if not case_is_done(str(ALL_CASES[idx]["name"]))]

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"BANK_ROOT_SUFFIX={bank_root_suffix!r}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"SECONDARY_FILTER_MIN_MEAN_PATCH={common_env['SECONDARY_FILTER_MIN_MEAN_PATCH']}")
    print(f"SECONDARY_FILTER_BOX_SCORE_MIN={common_env['SECONDARY_FILTER_BOX_SCORE_MIN']}")
    print(f"SOFTPATCH_KEEP_RATIO={common_env['SOFTPATCH_KEEP_RATIO']}")
    print(f"SOFTPATCH_KNN_K={common_env['SOFTPATCH_KNN_K']}")
    print(f"SOFTPATCH_SCORE_POWER={common_env['SOFTPATCH_SCORE_POWER']}")
    print(f"PATCHCORE_TOPK={common_env['PATCHCORE_TOPK']}")
    print(f"PATCHCORE_REWEIGHT_LAMBDA={common_env['PATCHCORE_REWEIGHT_LAMBDA']}")
    print(f"THRESHOLD_NORMAL_STD_MULT={common_env['THRESHOLD_NORMAL_STD_MULT']}")
    print(f"THRESHOLD_IMAGE_STD_MULT={common_env['THRESHOLD_IMAGE_STD_MULT']}")
    print(f"THRESHOLD_MIN_FLOOR_DELTA={common_env['THRESHOLD_MIN_FLOOR_DELTA']}")
    print(f"TOTAL_CASES={len(ALL_CASES)}")
    print(f"PARALLEL_SPLIT={len(first_half)}+{len(second_half)}")
    print(f"PENDING_SPLIT={len(pending_first_half)}+{len(pending_second_half)}")
    print(f"MAX_RESTARTS_PER_RUNNER={max_restarts}")
    print(f"LOG_DIR={log_dir}")
    print(
        "CATEGORY_TOTALS="
        f"cavities:{category_totals['cavities']}, "
        f"utilities:{category_totals['utilities']}, "
        f"normal_auc:{category_totals['normal_auc']}"
    )

    for bank_tag in sorted({str(case["bank_tag"]) for case in ALL_CASES}):
        bank_suffix = case_bank_suffix(bank_root_suffix, bank_tag)
        print(f"BANK[{bank_tag}]={bank_file_path(bank_suffix)}")

    build_bank_mode = env_str("BUILD_BANK", "auto").lower()
    print(f"BUILD_BANK_MODE={build_bank_mode}")

    pending_indices = pending_first_half + pending_second_half
    required_bank_tags = sorted({str(ALL_CASES[idx]["bank_tag"]) for idx in pending_indices})
    for bank_tag in required_bank_tags:
        bank_suffix = case_bank_suffix(bank_root_suffix, bank_tag)
        should_build = resolve_build_bank(build_bank_mode, bank_suffix)
        print(f"BUILD_BANK[{bank_tag}]={int(should_build)}")
        if not should_build:
            continue
        env = os.environ.copy()
        env.update(common_env)
        env["BANK_SUFFIX"] = bank_suffix
        env["USE_SOFTPATCH_BANK_FILTER"] = "1" if bank_tag == "softpatch" else "0"
        cmd = [PYTHON, str(BASE_DIR / "experiments" / "01_build_feature_bank_v26.py")]
        print(">>>", " ".join(cmd), f"[build bank:{bank_tag}]")
        subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)

    runners: list[dict] = []
    if pending_first_half:
        proc_a, log_a = spawn_runner(pending_first_half, "0", "V26 parallel part A", common_env, log_dir, 0)
        runners.append({
            "title": "V26 parallel part A",
            "case_indices": pending_first_half,
            "proc": proc_a,
            "log_path": log_a,
            "restart_count": 0,
            "max_restarts": max_restarts,
            "last_failure": "",
        })
    else:
        print("[resume] skip V26 parallel part A: all assigned cases already have evaluation reports")

    if pending_second_half:
        proc_b, log_b = spawn_runner(pending_second_half, "0", "V26 parallel part B", common_env, log_dir, 0)
        runners.append({
            "title": "V26 parallel part B",
            "case_indices": pending_second_half,
            "proc": proc_b,
            "log_path": log_b,
            "restart_count": 0,
            "max_restarts": max_restarts,
            "last_failure": "",
        })
    else:
        print("[resume] skip V26 parallel part B: all assigned cases already have evaluation reports")

    if not runners:
        print("\nAll pending v26 parallel cases are already completed.")
        return 0

    try:
        failures = monitor_parallel_runs(runners, category_totals, common_env, log_dir, poll_interval=1.0)
    finally:
        for runner in runners:
            close_proc_log(runner["proc"])

    final_rc = 0
    for runner in runners:
        rc = runner["proc"].wait()
        if rc != 0 and final_rc == 0:
            final_rc = rc

    print("\n=== FINAL LOG FILES ===")
    for runner in runners:
        print(runner["log_path"])

    if failures:
        print("\n=== RESUME HINT ===")
        print("Some runners still failed after auto-restart. Re-run 08_run_v26_parallel_sweep.py and it will continue unfinished cases.")
        raise SystemExit(final_rc or 1)

    print("\nV26 parallel sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
