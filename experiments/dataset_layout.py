"""
Central dataset paths for V4/V5 experiments.

- enhanced: legacy local pipeline (augmented_* eval, V4 bank from intact / V5 from augmented_intact).
- open_source | real_world | synthetic: expect data under ``<BASE_DIR>/data_paper/<mode>/``
  per paper_alignment_notes.md §5. Each mode keeps the same inference category keys
  (cavities, utilities, normal_auc) so 02/03 loops stay unchanged.

Override with env ``RES_SAM_DATASET_MODE`` (e.g. open_source).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

DATASET_ENHANCED = "enhanced"
DATASET_OPEN_SOURCE = "open_source"
DATASET_REAL_WORLD = "real_world"
DATASET_SYNTHETIC = "synthetic"

VALID_DATASET_MODES = (
    DATASET_ENHANCED,
    DATASET_OPEN_SOURCE,
    DATASET_REAL_WORLD,
    DATASET_SYNTHETIC,
)


def resolve_dataset_mode(config: Optional[Dict[str, Any]] = None) -> str:
    env = (os.environ.get("RES_SAM_DATASET_MODE") or "").strip().lower()
    if env:
        if env not in VALID_DATASET_MODES:
            raise ValueError(
                f"RES_SAM_DATASET_MODE={env!r} invalid; use one of {VALID_DATASET_MODES}"
            )
        return env
    if config:
        m = (config.get("dataset_mode") or DATASET_ENHANCED).strip().lower()
        if m not in VALID_DATASET_MODES:
            raise ValueError(f"dataset_mode={m!r} invalid; use one of {VALID_DATASET_MODES}")
        return m
    return DATASET_ENHANCED


def _voc_anno(subdir: str) -> str:
    return os.path.join(subdir, "annotations", "VOC_XML_format")


def _first_existing(*candidates: str) -> Optional[str]:
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _require_dir(path: str, context: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{context}\nMissing directory: {path}\n"
            f"Create it per paper_alignment_notes.md §5 (data_paper layout) or use dataset_mode='enhanced'."
        )


def build_layout(
    base_dir: str,
    dataset_mode: str,
    *,
    bank_variant: str = "v4",
) -> Dict[str, Any]:
    """
    Returns path bundle for experiments. bank_variant: 'v4' | 'v5' selects enhanced normal source.
    """
    gpr = os.path.join(base_dir, "data", "GPR_data")
    paper = os.path.join(base_dir, "data_paper")
    mode = dataset_mode

    if mode == DATASET_ENHANCED:
        if bank_variant == "v5":
            normal_sources = {
                "augmented_intact": os.path.join(gpr, "augmented_intact"),
            }
        else:
            normal_sources = {
                "intact": os.path.join(gpr, "intact"),
            }
        test_data_dirs = {
            "cavities": os.path.join(gpr, "augmented_cavities"),
            "utilities": os.path.join(gpr, "augmented_utilities"),
            "normal_auc": os.path.join(gpr, "augmented_intact"),
        }
        annotation_dirs = {
            "cavities": _voc_anno(os.path.join(gpr, "augmented_cavities")),
            "utilities": _voc_anno(os.path.join(gpr, "augmented_utilities")),
        }
        suffix = ""
    else:
        root = os.path.join(paper, mode)
        _require_dir(root, f"dataset_mode={mode!r} requires a paper-style data root.")

        if mode == DATASET_OPEN_SOURCE:
            n_normal = os.path.join(root, "intact")
            n_cav = os.path.join(root, "cavities")
            n_util = _first_existing(
                os.path.join(root, "pipes"),
                os.path.join(root, "utilities"),
            )
        elif mode == DATASET_REAL_WORLD:
            n_normal = os.path.join(root, "normal")
            n_cav = os.path.join(root, "cavities")
            n_util = _first_existing(
                os.path.join(root, "pipelines"),
                os.path.join(root, "pipes"),
                os.path.join(root, "utilities"),
            )
        else:  # synthetic
            n_normal = _first_existing(
                os.path.join(root, "normal"),
                os.path.join(root, "intact"),
            )
            n_cav = os.path.join(root, "cavities")
            n_util = _first_existing(
                os.path.join(root, "pipelines"),
                os.path.join(root, "pipes"),
                os.path.join(root, "metal_pipelines"),
                os.path.join(root, "utilities"),
            )

        if not n_util:
            raise FileNotFoundError(
                f"dataset_mode={mode!r}: could not find pipes/utilities/pipelines directory under {root}"
            )

        if not n_normal:
            raise FileNotFoundError(
                f"dataset_mode={mode!r}: need normal/ or intact/ under {root} for normal_auc + feature bank."
            )

        _require_dir(n_normal, f"dataset_mode={mode!r}: normal/intact folder")
        _require_dir(n_cav, f"dataset_mode={mode!r}: cavities folder")
        _require_dir(n_util, f"dataset_mode={mode!r}: utilities/pipes folder")

        normal_sources = {"normal": n_normal}
        test_data_dirs = {
            "cavities": n_cav,
            "utilities": n_util,
            "normal_auc": n_normal,
        }
        annotation_dirs = {
            "cavities": _voc_anno(n_cav),
            "utilities": _voc_anno(n_util),
        }
        for cat, ad in ("cavities", annotation_dirs["cavities"]), (
            "utilities",
            annotation_dirs["utilities"],
        ):
            if not os.path.isdir(ad):
                raise FileNotFoundError(
                    f"dataset_mode={mode!r}: need VOC annotations at {ad} for category {cat!r}."
                )
        suffix = f"_{mode}"

    ver = bank_variant.lower()
    fb_dir = os.path.join(base_dir, "outputs", f"feature_banks_{ver}{suffix}")
    pred_dir = os.path.join(base_dir, "outputs", f"predictions_{ver}{suffix}")
    ckpt_dir = os.path.join(base_dir, "outputs", f"checkpoints_{ver}{suffix}")
    vis_dir = os.path.join(base_dir, "outputs", f"visualizations_{ver}{suffix}")

    return {
        "dataset_mode": mode,
        "normal_data_sources": normal_sources,
        "test_data_dirs": test_data_dirs,
        "annotation_dirs": annotation_dirs,
        "feature_bank_dir": fb_dir,
        "feature_bank_path": os.path.join(fb_dir, f"feature_bank_{ver}.pth"),
        "metadata_path": os.path.join(fb_dir, "metadata.json"),
        "output_dir_predictions": pred_dir,
        "checkpoint_dir": ckpt_dir,
        "auto_predictions_file": os.path.join(pred_dir, f"auto_predictions_{ver}.json"),
        "click_predictions_file": os.path.join(pred_dir, f"click_predictions_{ver}.json"),
        "vis_output_dir": vis_dir,
        "analysis_output": os.path.join(
            vis_dir, f"04_evaluate_and_visualize_{ver}_report{suffix or ''}.md"
        ),
        "metrics_dir": os.path.join(base_dir, "outputs", f"metrics_{ver}{suffix}"),
        "clustering_output_file": os.path.join(
            base_dir, "outputs", f"metrics_{ver}{suffix}", f"clustering_{ver}.json"
        ),
        "clustering_checkpoint": os.path.join(
            base_dir, "outputs", f"checkpoints_{ver}{suffix}", f"checkpoint_clustering_{ver}.json"
        ),
        "layout_suffix": suffix,
    }


def apply_layout_to_config_01(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    mode = resolve_dataset_mode(config)
    lay = build_layout(base_dir, mode, bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = mode
    c["normal_data_sources"] = lay["normal_data_sources"]
    c["output_dir"] = lay["feature_bank_dir"]
    c["alignment_notes"] = (c.get("alignment_notes") or "") + f" | dataset_mode={mode}"
    return c


def apply_layout_to_config_02_03(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    mode = resolve_dataset_mode(config)
    lay = build_layout(base_dir, mode, bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = mode
    c["test_data_dirs"] = lay["test_data_dirs"]
    c["annotation_dirs"] = lay["annotation_dirs"]
    c["feature_bank_path"] = lay["feature_bank_path"]
    c["metadata_path"] = lay["metadata_path"]
    c["output_dir"] = lay["output_dir_predictions"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    c["annotation_dirs"] = dict(lay["annotation_dirs"])
    c["annotation_dirs"]["normal_auc"] = ""
    c["alignment_notes"] = (c.get("alignment_notes") or "") + f" | dataset_mode={mode}"
    return c


def apply_layout_to_config_04(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    mode = resolve_dataset_mode(config)
    lay = build_layout(base_dir, mode, bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = mode
    c["predictions_path"] = lay["auto_predictions_file"]
    c["click_predictions_path"] = lay["click_predictions_file"]
    c["vis_output_dir"] = lay["vis_output_dir"]
    c["analysis_output"] = lay["analysis_output"]
    c["alignment_notes"] = (c.get("alignment_notes") or "") + f" | dataset_mode={mode}"
    return c


def apply_layout_to_config_05(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["predictions_path"] = lay["auto_predictions_file"]
    c["test_data_dirs"] = {k: lay["test_data_dirs"][k] for k in ("cavities", "utilities")}
    c["annotation_dirs"] = {k: lay["annotation_dirs"][k] for k in ("cavities", "utilities")}
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c


def apply_layout_to_config_06(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["test_data_dirs"] = {k: lay["test_data_dirs"][k] for k in ("cavities", "utilities")}
    c["feature_bank_path"] = lay["feature_bank_path"]
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c


def apply_layout_to_config_07(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["test_data_dirs"] = lay["test_data_dirs"]
    c["annotation_dirs"] = dict(lay["annotation_dirs"])
    c["annotation_dirs"]["normal_auc"] = ""
    fb = dict(c.get("feature_bank_sources") or {})
    if fb:
        for k in list(fb.keys()):
            fb[k] = lay["feature_bank_path"]
    else:
        fb = {"intact": lay["feature_bank_path"]}
    c["feature_bank_sources"] = fb
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c
