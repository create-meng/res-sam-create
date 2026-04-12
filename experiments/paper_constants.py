"""
Paper-aligned numeric defaults shared by experiments 01–07.

Naming:
- beta_threshold: Eq.(9) β — anomaly likelihood vs feature bank (L2 distance scale).
  Same quantity as zhouxr6066/Res-SAM PatchRes/main.py `anomaly_threshold` and UI score cut (0.1).

- EVAL_DETECTION_IOU_THRESHOLD: main-text criterion for a "correct" detection (pred vs GT IoU).
  Not the same as β; do not mix in CONFIG or logs.

Preflight (faiss):
- preflight_faiss_or_raise(): import faiss + minimal IndexFlatL2 test; raises SystemExit(1) on failure.
- Default is STRICT: faiss must work. Skipping preflight is only allowed when you
  explicitly permit sklearn fallback (RES_SAM_ALLOW_SKLEARN_KNN=1) and also set
  RES_SAM_SKIP_FAISS_PREFLIGHT=1.
"""

from __future__ import annotations

import os
import sys

# Eq.(9) β — author public script / GUI use 0.1; paper gives symbol only.
DEFAULT_BETA_THRESHOLD = 0.1

# V10: data-driven calibrated β — p99 of Feature Bank internal 1-NN L2 distances.
# Rationale: beta=0.1 was tuned for the official code's normalized [0,1] score space.
# In raw L2 distance space (paper-faithful), p99 of normal-patch distances ≈ 0.183
# gives ~1% false-positive rate on normal patches vs 24.8% with beta=0.1.
# This is a valid "predefined threshold" in the paper's sense (Eq.(9): β is predefined).
DEFAULT_BETA_THRESHOLD_V10 = 0.183

# V11: split coarse/fine beta.
# coarse_beta: low threshold → keep more candidate regions (don't miss true anomalies)
# fine_beta:   high threshold → strict patch-level filtering (reduce FP)
# Diagnosis: V10 raised beta to 0.183 uniformly → coarse stage discarded 136 regions (vs 87 in V9)
# including true anomaly regions → TP dropped from 12 to 7.
DEFAULT_COARSE_BETA_V11 = 0.05   # 宽松粗筛：几乎不丢region，让真异常region进入细筛
DEFAULT_FINE_BETA_V11   = 0.183  # 严格细筛：p99校准值，减少FP

# ~page 4: "IoU with the ground truth exceeds a threshold of 0.5" for accuracy / Table-style eval.
EVAL_DETECTION_IOU_THRESHOLD = 0.5

# Conservative / paper-faithful defaults for optional engineering knobs (see paper_alignment_notes.md §6.1):
# - Keep max_candidates_per_image and min_region_area unset (None) in CONFIG unless doing ablations.
# - Do not enable data-driven beta calibration in mainline scripts unless explicitly documented.


def preflight_faiss_or_raise() -> None:
    """
    Verify faiss loads and a tiny L2 index works. Call at the start of each experiment script 01–07.

    Default is STRICT: faiss must work.

    To skip preflight you must explicitly allow sklearn fallback:
    - RES_SAM_ALLOW_SKLEARN_KNN=1
    - RES_SAM_SKIP_FAISS_PREFLIGHT=1
    """
    allow_sklearn = (os.environ.get("RES_SAM_ALLOW_SKLEARN_KNN") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    skip = (os.environ.get("RES_SAM_SKIP_FAISS_PREFLIGHT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if skip and allow_sklearn:
        print(
            "[preflight_faiss] SKIP (RES_SAM_SKIP_FAISS_PREFLIGHT): "
            "sklearn fallback is permitted (RES_SAM_ALLOW_SKLEARN_KNN=1).",
            flush=True,
        )
        return
    if skip and (not allow_sklearn):
        print(
            "[preflight_faiss] NOTE: RES_SAM_SKIP_FAISS_PREFLIGHT is set but ignored because "
            "RES_SAM_ALLOW_SKLEARN_KNN is not enabled. Default is strict-faiss.",
            flush=True,
        )

    try:
        import faiss
        import numpy as np

        d = 8
        rng = np.random.RandomState(0)
        xb = rng.random((16, d)).astype(np.float32)
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        dists, _ = index.search(xb[:3], 1)
        if dists.shape != (3, 1):
            raise RuntimeError(f"unexpected search output shape {dists.shape}")
        ver = getattr(faiss, "__version__", "unknown")
    except SystemExit:
        raise
    except Exception as e:
        print(
            "[preflight_faiss] FAILED: import or minimal faiss IndexFlatL2 test.\n"
            "  Fix: conda install -n <env> -c pytorch faiss-cpu  (or pip faiss-cpu for your Python).\n"
            "  Windows DLL errors: reinstall faiss or install MSVC runtime.\n"
            "  If you really want to run without faiss, set RES_SAM_ALLOW_SKLEARN_KNN=1 "
            "(and optionally RES_SAM_SKIP_FAISS_PREFLIGHT=1).\n"
            f"  Error: {e}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from e
    print(f"[preflight_faiss] OK (faiss version={ver})", flush=True)
