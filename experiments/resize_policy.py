"""
Shared image resize policy for V4/V5 experiment scripts.

The paper mentions concrete sizes for some real-world GPR imagery (e.g. 369x369 and 615x369)
but does not give one universal “resize everything to HxW” rule for every reproduction path.

Policy:
- fixed: always resize to config["image_size"] (H, W). Legacy / comparable to older runs.
- voc_annotation: if VOC XML is available, resize to XML (height, width); otherwise keep native
  file resolution (no resize). Aligns bboxes with the annotation coordinate system without
  inventing a global square resize.
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
    policy = config.get("resize_policy", RESIZE_POLICY_VOC_ANNOTATION)
    if policy == RESIZE_POLICY_FIXED:
        return config.get("image_size")
    if policy == RESIZE_POLICY_VOC_ANNOTATION:
        if voc_size_record and voc_size_record.get("width") and voc_size_record.get("height"):
            return (int(voc_size_record["height"]), int(voc_size_record["width"]))
        return None
    return config.get("image_size")
