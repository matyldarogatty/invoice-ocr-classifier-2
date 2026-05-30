"""Unit tests for layout feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from layout_features import (  # noqa: E402
    LAYOUT_FEATURE_COLUMNS,
    LINE_NO_COLUMN,
    active_layout_columns,
    compute_layout_features,
    extract_line_bbox_norm,
    validate_layout_features,
)


class _Word:
    def __init__(self, geometry):
        self.geometry = geometry


class _Line:
    def __init__(self, words):
        self.words = words


def _line_from_boxes(boxes):
    words = [_Word(geom) for geom in boxes]
    return _Line(words)


def test_layout_feature_columns_count_and_names():
    assert len(LAYOUT_FEATURE_COLUMNS) == 10
    assert LAYOUT_FEATURE_COLUMNS[-1] == LINE_NO_COLUMN
    assert "bbox_x_min_norm" in LAYOUT_FEATURE_COLUMNS
    assert "bbox_area_norm" in LAYOUT_FEATURE_COLUMNS


def test_extract_line_bbox_norm_basic():
    line = _line_from_boxes([((0.1, 0.2), (0.5, 0.4))])
    bbox = extract_line_bbox_norm(line)
    assert bbox is not None
    assert bbox["bbox_x_min_norm"] == pytest.approx(0.1)
    assert bbox["bbox_y_max_norm"] == pytest.approx(0.4)
    assert bbox["bbox_width_norm"] == pytest.approx(0.4)
    assert bbox["bbox_height_norm"] == pytest.approx(0.2)
    assert bbox["bbox_area_norm"] == pytest.approx(0.08)


def test_extract_line_bbox_norm_empty_line():
    line = _Line([])
    assert extract_line_bbox_norm(line) is None


def test_compute_layout_features_line_no_norm():
    line = _line_from_boxes([((0.0, 0.0), (1.0, 0.1))])
    feats = compute_layout_features(line, line_idx=0, total_lines=5)
    assert feats is not None
    assert feats["line_no_norm"] == pytest.approx(0.0)

    feats_last = compute_layout_features(line, line_idx=4, total_lines=5)
    assert feats_last is not None
    assert feats_last["line_no_norm"] == pytest.approx(1.0)


def test_validate_layout_features_ok():
    line = _line_from_boxes([((0.1, 0.1), (0.3, 0.2))])
    feats = compute_layout_features(line, line_idx=1, total_lines=3)
    assert feats is not None
    validate_layout_features(feats)


def test_validate_layout_features_rejects_nan():
    feats = {col: 0.5 for col in LAYOUT_FEATURE_COLUMNS}
    feats["bbox_x_min_norm"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_layout_features(feats)


def test_validate_layout_features_rejects_out_of_range():
    line = _line_from_boxes([((0.1, 0.1), (0.3, 0.2))])
    feats = compute_layout_features(line, line_idx=0, total_lines=1)
    assert feats is not None
    feats["bbox_x_max_norm"] = 1.5
    with pytest.raises(ValueError, match="out of range"):
        validate_layout_features(feats)


def test_validate_layout_features_rejects_inverted_x():
    feats = {col: 0.5 for col in LAYOUT_FEATURE_COLUMNS}
    feats["bbox_x_min_norm"] = 0.8
    feats["bbox_x_max_norm"] = 0.2
    with pytest.raises(ValueError, match="bbox_x_min_norm"):
        validate_layout_features(feats)


def test_active_layout_columns_exclude_line_no():
    cols = active_layout_columns(exclude_line_no=True)
    assert len(cols) == 9
    assert LINE_NO_COLUMN not in cols
