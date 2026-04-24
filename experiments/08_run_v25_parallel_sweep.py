"""
V25 parallel quick sweep runner.

用途：
- 自动拆成两路并行运行
- 父进程按需只建一次 bank，再拆成前后两组
- 子进程日志分别写入文件，避免 tqdm 抢占同一终端
- 父进程定时读取 checkpoint，汇总显示两路进度
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
SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v25_quick_sweep.py"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints_v25"
PREDICTION_DIR = BASE_DIR / "outputs" / "predictions_v25"
LOG_ROOT_DIR = BASE_DIR / "outputs" / "parallel_logs_v25"


def env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


def build_common_env() -> dict[str, str]:
    return {
        "BANK_SUFFIX": env_str("BANK_SUFFIX", ""),
        "MAX_IMAGES_PER_CATEGORY": env_str("MAX_IMAGES_PER_CATEGORY", "30"),
    }


ALL_CASES = [
    ("base", 0),
    ("nfuse", 1),
]

CASE_TO_SUFFIX = {name: f"_{name}" for name, _ in ALL_CASES}

CATEGORY_DIRS = {
    "cavities": BASE_DIR / "data" / "GPR_data" / "augmented_cavities",
    "utilities": BASE_DIR / "data" / "GPR_data" / "augmented_utilities",
    "normal_auc": BASE_DIR / "data" / "GPR_data" / "augmented_intact",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v25{bank_suffix}" / f"feature_bank_v25{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v25" / "feature_bank_v25.pth"


def resolve_build_bank(common_env: dict[str, str], default_mode: str) -> bool:
    mode = env_str("BUILD_BANK", default_mode).lower()
    if mode == "1":
        return True
    if mode == "0":
        return False
    if mode == "auto":
        return not bank_file_path(common_env["BANK_SUFFIX"]).exists()
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
    return CHECKPOINT_DIR / f"checkpoint_auto_{category}{CASE_TO_SUFFIX[case_name]}.json"


def prediction_path(case_name: str) -> Path:
    return PREDICTION_DIR / f"auto_predictions_v25{CASE_TO_SUFFIX[case_name]}.json"


def report_path(case_name: str) -> Path:
    return PREDICTION_DIR / f"evaluation_report_v25{CASE_TO_SUFFIX[case_name]}.json"


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


def summarize_case(case_name: str, title: str, log_path: Path,
                   category_totals: dict[str, int], proc: subprocess.Popen | None,
                   restart_count: int) -> str:
    category_chunks: list[str] = []
    completed_categories = 0
    seen_checkpoint = False
    latest_category = ""
    latest_timestamp = ""

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
            ts = str(info.get("timestamp", "") or "")
            if ts and ts >= latest_timestamp:
                latest_timestamp = ts
                latest_category = category
        if completed:
            completed_categories += 1
        short_name = {"cavities": "cav", "utilities": "util", "normal_auc": "normal"}[category]
        category_chunks.append(f"{short_name} {processed}/{total}")

    if report_path(case_name).exists():
        stage = "done"
    elif prediction_path(case_name).exists() and completed_categories == 3:
        stage = "evaluating"
    elif seen_checkpoint:
        stage = f"running:{latest_category or 'unknown'}"
    elif proc is not None and proc.poll() is None:
        stage = "starting"
    else:
        stage = "pending"

    return f"[{title}] {case_name} | {stage} | restart={restart_count} | {' | '.join(category_chunks)} | log={log_path.name}"


def open_log_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)


def spawn_runner(case_indices: list[int], build_bank: str, title: str,
                 common_env: dict[str, str], log_dir: Path, restart_count: int) -> tuple[subprocess.Popen, Path]:
    env = os.environ.copy()
    env.update(common_env)
    env["BUILD_BANK"] = build_bank
    env["V25_CASE_INDICES"] = ",".join(str(i) for i in case_indices)
    env["V25_SWEEP_TITLE"] = title

    first_case_name = ALL_CASES[case_indices[0]][0]
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
    print(f"case={runner['case_name']}")
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
        lines = [
            summarize_case(
                runner["case_name"], runner["title"], runner["log_path"],
                category_totals, runner["proc"], runner["restart_count"]
            )
            for runner in runners
        ]
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
            if runner["case_name"] in failures:
                continue

            status = report_runner_failure(runner)
            if runner["restart_count"] < runner["max_restarts"] and not case_is_done(runner["case_name"]):
                print(
                    f"[monitor] restarting {runner['title']} "
                    f"({runner['restart_count'] + 1}/{runner['max_restarts']}) from checkpoint"
                )
                restart_runner(runner, common_env, log_dir)
                last_snapshot = ""
                continue

            failures[runner["case_name"]] = status

        if all(
            case_is_done(runner["case_name"]) or
            (runner["proc"].poll() is not None and runner["case_name"] in failures) or
            (runner["proc"].poll() == 0)
            for runner in runners
        ):
            break
        time.sleep(poll_interval)
    return failures


def main() -> int:
    common_env = build_common_env()
    bank_suffix = common_env["BANK_SUFFIX"]
    category_totals = build_category_totals(common_env)
    log_dir = LOG_ROOT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    max_restarts = int(env_str("MAX_RESTARTS_PER_RUNNER", "2"))

    first_half = list(range(0, len(ALL_CASES) // 2))
    second_half = list(range(len(ALL_CASES) // 2, len(ALL_CASES)))
    pending_first_half = [idx for idx in first_half if not case_is_done(ALL_CASES[idx][0])]
    pending_second_half = [idx for idx in second_half if not case_is_done(ALL_CASES[idx][0])]

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"BANK_SUFFIX={bank_suffix!r}")
    print(f"BANK_PATH={bank_file_path(bank_suffix)}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
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

    build_bank = resolve_build_bank(common_env, "auto")
    print(f"BUILD_BANK={int(build_bank)}")
    print(f"BUILD_BANK_MODE={env_str('BUILD_BANK', 'auto').lower()}")

    if build_bank:
        env = os.environ.copy()
        env.update(common_env)
        cmd = [PYTHON, str(BASE_DIR / "experiments" / "01_build_feature_bank_v25.py")]
        print(">>>", " ".join(cmd), "[V25 parallel build bank]")
        subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)
    else:
        print("\n=== SKIP FEATURE BANK BUILD: existing bank detected or BUILD_BANK=0 ===")

    runners: list[dict] = []
    if pending_first_half:
        proc_a, log_a = spawn_runner(pending_first_half, "0", "V25 parallel part A", common_env, log_dir, 0)
        runners.append({
            "title": "V25 parallel part A",
            "case_name": ALL_CASES[pending_first_half[0]][0],
            "case_indices": pending_first_half,
            "proc": proc_a,
            "log_path": log_a,
            "restart_count": 0,
            "max_restarts": max_restarts,
            "last_failure": "",
        })
    else:
        print("[resume] skip V25 parallel part A: all assigned cases already have evaluation reports")

    if pending_second_half:
        proc_b, log_b = spawn_runner(pending_second_half, "0", "V25 parallel part B", common_env, log_dir, 0)
        runners.append({
            "title": "V25 parallel part B",
            "case_name": ALL_CASES[pending_second_half[0]][0],
            "case_indices": pending_second_half,
            "proc": proc_b,
            "log_path": log_b,
            "restart_count": 0,
            "max_restarts": max_restarts,
            "last_failure": "",
        })
    else:
        print("[resume] skip V25 parallel part B: all assigned cases already have evaluation reports")

    if not runners:
        print("\nAll pending v25 parallel cases are already completed.")
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
        print("Some runners still failed after auto-restart. Re-run 08_run_v25_parallel_sweep.py and it will continue unfinished cases.")
        raise SystemExit(final_rc or 1)

    print("\nV25 parallel sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
