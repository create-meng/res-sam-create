"""
Paper-aligned numeric defaults shared by experiments 01–07.

Naming:
- beta_threshold: Eq.(9) β — anomaly likelihood vs feature bank (L2 distance scale).
  Same quantity as zhouxr6066/Res-SAM PatchRes/main.py `anomaly_threshold` and UI score cut (0.1).

- EVAL_DETECTION_IOU_THRESHOLD: main-text criterion for a "correct" detection (pred vs GT IoU).
  Not the same as β; do not mix in CONFIG or logs.
"""

# Eq.(9) β — author public script / GUI use 0.1; paper gives symbol only.
DEFAULT_BETA_THRESHOLD = 0.1

# ~page 4: "IoU with the ground truth exceeds a threshold of 0.5" for accuracy / Table-style eval.
EVAL_DETECTION_IOU_THRESHOLD = 0.5

# Conservative / paper-faithful defaults for optional engineering knobs (see paper_alignment_notes.md §6.1):
# - Keep max_candidates_per_image and min_region_area unset (None) in CONFIG unless doing ablations.
# - Do not enable data-driven beta calibration in mainline scripts unless explicitly documented.
