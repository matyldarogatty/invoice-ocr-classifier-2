"""Layout/bbox feature extraction from docTR OCR line geometry."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LAYOUT_FEATURE_COLUMNS: List[str] = [
    "bbox_x_min_norm",
    "bbox_y_min_norm",
    "bbox_x_max_norm",
    "bbox_y_max_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_area_norm",
    "line_no_norm",
]

LINE_NO_COLUMN = "line_no_norm"


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def extract_line_bbox_norm(line) -> Optional[Dict[str, float]]:
    """
    Return normalized line bbox (0-1) from docTR word.geometry.
    Returns None if the line has no words or invalid geometry.
    """
    xs: List[float] = []
    ys: List[float] = []
    for word in line.words:
        (x1, y1), (x2, y2) = word.geometry
        xs.extend([float(x1), float(x2)])
        ys.extend([float(y1), float(y2)])

    if not xs or not ys:
        return None

    x_min = _clip01(min(xs))
    x_max = _clip01(max(xs))
    y_min = _clip01(min(ys))
    y_max = _clip01(max(ys))

    if x_min > x_max or y_min > y_max:
        return None

    width = _clip01(x_max - x_min)
    height = _clip01(y_max - y_min)

    return {
        "bbox_x_min_norm": x_min,
        "bbox_y_min_norm": y_min,
        "bbox_x_max_norm": x_max,
        "bbox_y_max_norm": y_max,
        "bbox_width_norm": width,
        "bbox_height_norm": height,
        "bbox_center_x_norm": _clip01((x_min + x_max) / 2.0),
        "bbox_center_y_norm": _clip01((y_min + y_max) / 2.0),
        "bbox_area_norm": _clip01(width * height),
    }


def compute_layout_features(
    line,
    line_idx: int,
    total_lines: int,
) -> Optional[Dict[str, float]]:
    """
    Compute all layout features for one OCR line.
    line_idx is 0-based index among lines that passed the bbox size filter.
    """
    bbox = extract_line_bbox_norm(line)
    if bbox is None:
        return None

    if total_lines <= 0:
        return None

    line_no_norm = _clip01(line_idx / max(1, total_lines - 1))
    return {**bbox, "line_no_norm": line_no_norm}


def get_line_bbox_pixels(line, page) -> Tuple[int, int, int, int]:
    """Pixel bbox for cropping; envelope of all word geometries on the page."""
    xs: List[float] = []
    ys: List[float] = []
    for word in line.words:
        (x1, y1), (x2, y2) = word.geometry
        xs.extend([float(x1), float(x2)])
        ys.extend([float(y1), float(y2)])

    if not xs or not ys:
        raise ValueError("Line has no word geometry for pixel bbox")

    h, w = page.dimensions
    x1 = int(min(xs) * w)
    x2 = int(max(xs) * w)
    y1 = int(min(ys) * h)
    y2 = int(max(ys) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def validate_layout_features(features: Dict[str, float]) -> None:
    """Raise ValueError if layout features are invalid."""
    for col in LAYOUT_FEATURE_COLUMNS:
        if col not in features:
            raise ValueError(f"Missing layout feature: {col}")
        val = features[col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"NaN in layout feature: {col}")
        fv = float(val)
        if fv < 0.0 or fv > 1.0:
            raise ValueError(f"Layout feature {col} out of range [0,1]: {fv}")

    if features["bbox_x_min_norm"] > features["bbox_x_max_norm"]:
        raise ValueError("bbox_x_min_norm > bbox_x_max_norm")
    if features["bbox_y_min_norm"] > features["bbox_y_max_norm"]:
        raise ValueError("bbox_y_min_norm > bbox_y_max_norm")


def active_layout_columns(exclude_line_no: bool = False) -> List[str]:
    """Return layout columns used by the model (optionally without line_no_norm)."""
    if exclude_line_no:
        return [c for c in LAYOUT_FEATURE_COLUMNS if c != LINE_NO_COLUMN]
    return list(LAYOUT_FEATURE_COLUMNS)


def validate_layout_dataframe_columns(
    columns: Sequence[str],
    exclude_line_no: bool = False,
) -> None:
    """Ensure CSV/DataFrame has required layout columns."""
    required = active_layout_columns(exclude_line_no=exclude_line_no)
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing layout columns: {missing}")


def warn_if_invalid(features: Dict[str, float], context: str = "") -> None:
    """Log warning instead of raising (for export skip decisions)."""
    try:
        validate_layout_features(features)
    except ValueError as e:
        logging.getLogger("layout_features").warning(
            "Invalid layout features%s: %s", f" ({context})" if context else "", e
        )
