"""
Shared image resize policy for V6 experiment scripts.

Current V6 mainline policy:
- fixed: always resize to config["image_size"] (H, W).

Rationale:
- the paper does not define a new universal resize rule for reproduction;
- when the paper is silent, V6 falls back to the author public-code default
  preprocessing convention, i.e. fixed-size loading (369x369).

`voc_annotation` is kept only as a legacy helper and is not used by the
current V6 mainline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

RESIZE_POLICY_FIXED = "fixed"
RESIZE_POLICY_VOC_ANNOTATION = "voc_annotation"


def target_hw_for_preprocess(
    config: Dict[str, Any],
    voc_size_record: Optional[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    """
    Return (H, W) for PIL resize as (size[1], size[0]) in callers, or None for native resolution.

    voc_size_record:
        Any dict with integer-like "width" and "height" (VOC <size> block).
    """
    policy = config.get("resize_policy", RESIZE_POLICY_FIXED)
    if policy == RESIZE_POLICY_FIXED:
        return config.get("image_size")
    if policy == RESIZE_POLICY_VOC_ANNOTATION:
        if voc_size_record and voc_size_record.get("width") and voc_size_record.get("height"):
            return (int(voc_size_record["height"]), int(voc_size_record["width"]))
        return None
    return config.get("image_size")

