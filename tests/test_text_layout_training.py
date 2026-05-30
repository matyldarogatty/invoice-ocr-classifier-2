"""Tests for layout CSV validation and text training with layout features."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PROJECT = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from layout_features import LAYOUT_FEATURE_COLUMNS  # noqa: E402


def _mini_layout_df(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        x0 = 0.05 + (i % 5) * 0.05
        y0 = 0.05 + (i // 5) * 0.08
        rows.append(
            {
                "filename": f"inv_{i:04d}.png",
                "text": f"line text {i}",
                "label": i % 3,
                "invoice_id": f"inv_{i // 4}",
                "semantic_name": "OTHER",
                "bbox_x_min_norm": x0,
                "bbox_y_min_norm": y0,
                "bbox_x_max_norm": min(1.0, x0 + 0.2),
                "bbox_y_max_norm": min(1.0, y0 + 0.05),
                "bbox_width_norm": 0.2,
                "bbox_height_norm": 0.05,
                "bbox_center_x_norm": min(1.0, x0 + 0.1),
                "bbox_center_y_norm": min(1.0, y0 + 0.025),
                "bbox_area_norm": 0.01,
                "line_no_norm": (i % 4) / 3.0,
            }
        )
    return pd.DataFrame(rows)


def test_layout_csv_columns_and_ranges():
    df = _mini_layout_df()
    for col in LAYOUT_FEATURE_COLUMNS:
        assert col in df.columns
        assert df[col].isna().sum() == 0
        assert (df[col] >= 0.0).all()
        assert (df[col] <= 1.0).all()
    assert (df["bbox_x_min_norm"] <= df["bbox_x_max_norm"]).all()
    assert (df["bbox_y_min_norm"] <= df["bbox_y_max_norm"]).all()


def test_train_text_baseline_ignores_extra_columns(tmp_path: Path):
    df = _mini_layout_df(30)
    train = df.iloc[:20]
    val = df.iloc[20:25]
    test = df.iloc[25:30]
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "out_baseline"
    cmd = [
        sys.executable,
        str(SRC / "train_text_baseline.py"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--val-csv",
        str(tmp_path / "val.csv"),
        "--test-csv",
        str(tmp_path / "test.csv"),
        "--output-dir",
        str(out),
        "--model",
        "linear_svc",
        "--seed",
        "42",
    ]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_type"] == "text_ocr"
    assert metrics.get("layout_features_used") is False


def test_train_text_baseline_with_layout_features(tmp_path: Path):
    df = _mini_layout_df(30)
    train = df.iloc[:20]
    val = df.iloc[20:25]
    test = df.iloc[25:30]
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "out_layout"
    cmd = [
        sys.executable,
        str(SRC / "train_text_baseline.py"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--val-csv",
        str(tmp_path / "val.csv"),
        "--test-csv",
        str(tmp_path / "test.csv"),
        "--output-dir",
        str(out),
        "--model",
        "linear_svc",
        "--seed",
        "42",
        "--use-layout-features",
    ]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_type"] == "text_ocr+layout"
    assert metrics["layout_features_used"] is True
    assert (out / "model.joblib").is_file()


def test_train_text_baseline_layout_without_line_no(tmp_path: Path):
    df = _mini_layout_df(30)
    train = df.iloc[:20]
    val = df.iloc[20:25]
    test = df.iloc[25:30]
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "out_layout_no_line_no"
    cmd = [
        sys.executable,
        str(SRC / "train_text_baseline.py"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--val-csv",
        str(tmp_path / "val.csv"),
        "--test-csv",
        str(tmp_path / "test.csv"),
        "--output-dir",
        str(out),
        "--model",
        "linear_svc",
        "--seed",
        "42",
        "--use-layout-features",
        "--exclude-line-no-feature",
    ]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["line_no_feature_used"] is False
    assert "line_no_norm" not in config["layout_feature_columns"]


@pytest.mark.skipif(
    not (PROJECT / "data" / "labels_synthetic_with_layout.csv").is_file(),
    reason="Layout CSV not exported yet",
)
def test_real_layout_csv_validation():
    from validate_layout_csv import validate_layout_csv

    path = PROJECT / "data" / "labels_synthetic_with_layout.csv"
    result = validate_layout_csv(path)
    assert result["ok"] is True
    assert result["layout_columns_present"] == 10
    assert result["nan_count"] == 0
