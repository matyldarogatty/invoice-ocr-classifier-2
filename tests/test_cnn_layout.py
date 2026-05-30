"""Tests for CNN + layout features (dataset, model, training smoke)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
PROJECT = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import IMG_SIZE  # noqa: E402
from invoice_dataset import InvoiceDataset  # noqa: E402
from layout_features import LAYOUT_FEATURE_COLUMNS  # noqa: E402
from model import InvoiceCNN, InvoiceCNNWithLayout  # noqa: E402


def _mini_layout_df(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        x0 = 0.05 + (i % 4) * 0.05
        y0 = 0.05 + (i // 4) * 0.08
        rows.append(
            {
                "filename": f"line_{i:04d}.png",
                "text": f"line {i}",
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


def _write_dummy_images(df: pd.DataFrame, images_dir: Path) -> None:
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    for fn in df["filename"]:
        Image.new("RGB", (64, 32), color=(200, 200, 200)).save(images_dir / fn)


def test_dataset_without_layout_returns_image_label(tmp_path: Path):
    df = _mini_layout_df(4)
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    ds = InvoiceDataset(
        dataframe=df,
        images_dir=str(img_dir),
        use_layout_features=False,
    )
    image, label = ds[0]
    assert image.shape == (1, IMG_SIZE, IMG_SIZE)
    assert isinstance(label, int)


def test_dataset_with_layout_returns_three_tuple(tmp_path: Path):
    df = _mini_layout_df(4)
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    ds = InvoiceDataset(
        dataframe=df,
        images_dir=str(img_dir),
        use_layout_features=True,
    )
    image, layout, label = ds[0]
    assert image.shape == (1, IMG_SIZE, IMG_SIZE)
    assert layout.shape == (10,)
    assert layout.dtype == torch.float32
    assert float(layout.min()) >= 0.0
    assert float(layout.max()) <= 1.0
    assert isinstance(label, int)


def test_dataset_exclude_line_no_shape(tmp_path: Path):
    df = _mini_layout_df(4)
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    ds = InvoiceDataset(
        dataframe=df,
        images_dir=str(img_dir),
        use_layout_features=True,
        exclude_line_no_feature=True,
    )
    _, layout, _ = ds[0]
    assert layout.shape == (9,)


def test_dataset_missing_layout_columns_raises(tmp_path: Path):
    df = _mini_layout_df(2)[["filename", "text", "label", "invoice_id", "semantic_name"]]
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    with pytest.raises(ValueError, match="Missing layout columns"):
        InvoiceDataset(
            dataframe=df,
            images_dir=str(img_dir),
            use_layout_features=True,
        )


def test_invoice_cnn_forward():
    model = InvoiceCNN(num_classes=5)
    x = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    out = model(x)
    assert out.shape == (4, 5)


def test_invoice_cnn_with_layout_forward_10():
    model = InvoiceCNNWithLayout(num_classes=5, layout_dim=10)
    x = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    layout = torch.rand(4, 10)
    out = model(x, layout)
    assert out.shape == (4, 5)


def test_invoice_cnn_with_layout_forward_9():
    model = InvoiceCNNWithLayout(num_classes=5, layout_dim=9)
    x = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
    layout = torch.rand(4, 9)
    out = model(x, layout)
    assert out.shape == (4, 5)


def test_train_cnn_without_layout_on_layout_csv(tmp_path: Path):
    df = _mini_layout_df(12)
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    train = df.iloc[:8]
    val = df.iloc[8:10]
    test = df.iloc[10:12]
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "out_cnn"
    cmd = [
        sys.executable,
        str(SRC / "train.py"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--val-csv",
        str(tmp_path / "val.csv"),
        "--test-csv",
        str(tmp_path / "test.csv"),
        "--images-dir",
        str(img_dir),
        "--output-dir",
        str(out),
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--seed",
        "42",
    ]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_type"] == "image"
    assert metrics.get("layout_features_used") is False


def test_train_cnn_with_layout_smoke(tmp_path: Path):
    df = _mini_layout_df(12)
    img_dir = tmp_path / "imgs"
    _write_dummy_images(df, img_dir)
    train = df.iloc[:8]
    val = df.iloc[8:10]
    test = df.iloc[10:12]
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "out_cnn_layout"
    cmd = [
        sys.executable,
        str(SRC / "train.py"),
        "--train-csv",
        str(tmp_path / "train.csv"),
        "--val-csv",
        str(tmp_path / "val.csv"),
        "--test-csv",
        str(tmp_path / "test.csv"),
        "--images-dir",
        str(img_dir),
        "--output-dir",
        str(out),
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--use-layout-features",
    ]
    subprocess.run(cmd, cwd=str(PROJECT), check=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_type"] == "image_crop+layout"
    assert metrics["layout_features_used"] is True
    assert metrics["model_key"] == "cnn_layout"
    assert (out / "model.pth").is_file()
