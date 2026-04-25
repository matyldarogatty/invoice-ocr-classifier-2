import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight as sk_class_weight
from torch.utils.data import DataLoader, random_split

from config import IMG_SIZE, LABELS, NUM_CLASSES
from invoice_dataset import InvoiceDataset
from metrics_reporting import (
    compute_split_metrics,
    confusion_matrix_to_csv,
    save_json,
)
from model import InvoiceCNN

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg in ("cpu", "cuda"):
        if arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return arg
    raise ValueError("device must be one of: auto, cpu, cuda")


def _check_paths_exist(
    train_csv: Path, val_csv: Path, test_csv: Path, images_dir: Path
) -> None:
    for p, name in [
        (train_csv, "train-csv"),
        (val_csv, "val-csv"),
        (test_csv, "test-csv"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {name}: {p}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images-dir is not a directory: {images_dir}")


def _check_required_columns(path: Path) -> None:
    d = pd.read_csv(path, nrows=0)
    cols = [str(c).strip() for c in d.columns]
    for c in ("filename", "label"):
        if c not in cols:
            raise ValueError(f"{path} is missing required column: {c!r}")


def _validate_no_empty_labels(path: Path) -> None:
    d = pd.read_csv(path, usecols=["label"])
    if d["label"].isna().any():
        raise ValueError(f"Empty label values in {path}")


def _check_images(
    paths: List[Path], images_dir: Path, limit_msg: int = 5
) -> None:
    missing: List[str] = []
    for pth in paths:
        d = pd.read_csv(pth)
        d.columns = [str(c).strip() for c in d.columns]
        for _, row in d.iterrows():
            fn = row["filename"]
            full = images_dir / str(fn)
            if not full.is_file():
                missing.append(str(fn))
    if missing:
        sample = ", ".join(missing[:limit_msg])
        extra = f" and {len(missing) - limit_msg} more" if len(missing) > limit_msg else ""
        raise FileNotFoundError(
            f"Missing image file(s) under {images_dir} (e.g. {sample}{extra})"
        )


def _build_class_weights(
    train_csv: Path, num_classes: int, use_weights: bool, device: str
) -> Optional[torch.Tensor]:
    if not use_weights:
        return None
    d = pd.read_csv(train_csv)
    d.columns = [str(c).strip() for c in d.columns]
    y = d["label"].values.astype(int)
    unique = np.unique(y)
    cw = sk_class_weight.compute_class_weight("balanced", classes=unique, y=y)
    w = np.ones(num_classes, dtype=np.float32)
    for c, wgt in zip(unique, cw):
        w[c] = wgt
    return torch.from_numpy(w).to(device)


@torch.no_grad()
def _predict_loader(
    loader: DataLoader,
    model: nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_pred)


def _train_experiment_mode(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _check_paths_exist(
        Path(args.train_csv), Path(args.val_csv), Path(args.test_csv), Path(args.images_dir)
    )
    for p in (Path(args.train_csv), Path(args.val_csv), Path(args.test_csv)):
        _check_required_columns(p)
        _validate_no_empty_labels(p)
    tds = len(pd.read_csv(args.train_csv))
    vds = len(pd.read_csv(args.val_csv))
    sds = len(pd.read_csv(args.test_csv))
    if tds == 0 or vds == 0 or sds == 0:
        raise ValueError("train, val, and test CSVs must be non-empty.")
    _check_images(
        [Path(args.train_csv), Path(args.val_csv), Path(args.test_csv)],
        Path(args.images_dir),
    )

    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    train_set = InvoiceDataset(
        str(args.train_csv), str(args.images_dir), strict=True
    )
    val_set = InvoiceDataset(str(args.val_csv), str(args.images_dir), strict=True)
    test_set = InvoiceDataset(
        str(args.test_csv), str(args.images_dir), strict=True
    )

    train_loader = DataLoader(
        train_set, batch_size=int(args.batch_size), shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=int(args.batch_size), shuffle=False
    )

    model = InvoiceCNN().to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate)
    )
    ce_weight = _build_class_weights(
        Path(args.train_csv), NUM_CLASSES, bool(args.use_class_weights), device
    )
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    report_labels = list(range(NUM_CLASSES))

    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out_logits = model(x)
            loss = criterion(out_logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        n_batches = max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1}/{int(args.epochs)} — train loss: {total_loss / n_batches:.4f}"
        )

    y_val_t, y_val_p = _predict_loader(val_loader, model, device)
    y_test_t, y_test_p = _predict_loader(test_loader, model, device)
    if y_val_t.size == 0 or y_test_t.size == 0:
        raise RuntimeError("Empty val or test predictions.")

    val_metrics = compute_split_metrics(
        y_val_t, y_val_p, num_classes=NUM_CLASSES, labels=report_labels
    )
    test_metrics = compute_split_metrics(
        y_test_t, y_test_p, num_classes=NUM_CLASSES, labels=report_labels
    )
    val_cm = confusion_matrix(
        y_val_t, y_val_p, labels=np.arange(NUM_CLASSES)
    )
    test_cm = confusion_matrix(
        y_test_t, y_test_p, labels=np.arange(NUM_CLASSES)
    )
    val_rep = classification_report(
        y_val_t,
        y_val_p,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )
    test_rep = classification_report(
        y_test_t,
        y_test_p,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )

    config_payload: Dict[str, Any] = {
        "model_name": "InvoiceCNN",
        "input_type": "image",
        "input_shape": f"1x{IMG_SIZE}x{IMG_SIZE} (grayscale after transform)",
        "train_csv": str(Path(args.train_csv).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()),
        "test_csv": str(Path(args.test_csv).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "output_dir": str(out.resolve()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "device": device,
        "use_class_weights": bool(args.use_class_weights),
        "num_classes": NUM_CLASSES,
    }
    save_json(out / "config.json", config_payload)
    save_json(
        out / "metrics.json",
        {
            "model_name": "InvoiceCNN",
            "input_type": "image",
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    save_json(out / "classification_report_val.json", val_rep)
    save_json(out / "classification_report_test.json", test_rep)
    confusion_matrix_to_csv(out / "confusion_matrix_val.csv", val_cm, label_names=LABELS)
    confusion_matrix_to_csv(
        out / "confusion_matrix_test.csv", test_cm, label_names=LABELS
    )
    torch.save(model.state_dict(), out / "model.pth")

    print("=== Val metrics ===", val_metrics)
    print("=== Test metrics ===", test_metrics)
    print(f"Saved model to {out / 'model.pth'}")


def _legacy_80_20() -> None:
    """Match previous one-file behavior: 80% train, 20% val, no test; outputs under project dirs."""
    csv_p = PROJECT_DIR / "data" / "labels.csv"
    images = PROJECT_DIR / "data" / "images"
    if not csv_p.is_file():
        raise FileNotFoundError(f"Legacy mode expects {csv_p}")
    if not images.is_dir():
        raise FileNotFoundError(f"Legacy mode expects {images}")
    out_debug = PROJECT_DIR / "output" / "debug"
    out_models = PROJECT_DIR / "models"
    out_debug.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    dataset = InvoiceDataset(str(csv_p), str(images), strict=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size < 1 or val_size < 1:
        raise ValueError("Not enough rows for 80/20 split in legacy mode.")
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    device = "cpu"
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = InvoiceCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        y_t, y_p = _predict_loader(val_loader, model, device)
        cm = confusion_matrix(
            y_t, y_p, labels=np.arange(NUM_CLASSES)
        )
        acc = (y_t == y_p).mean() if y_t.size else 0.0
        print(
            f"Epoch {epoch + 1}: loss={total_loss / max(1, len(train_loader)):.4f} val_acc={acc:.4f}"
        )
        print(cm)
    torch.save(
        model.state_dict(), out_models / "invoice_cnn.pth"
    )
    print("Legacy: saved to models/invoice_cnn.pth and wrote epoch logs only.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train InvoiceCNN on image line crops (split CSVs or legacy 80/20)."
    )
    p.add_argument(
        "--legacy-80-20",
        action="store_true",
        help="Use data/labels.csv + data/images with random 80/20, no test set, save models/invoice_cnn.pth (old behavior).",
    )
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use balanced class weights from the training set for CrossEntropyLoss.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.legacy_80_20:
        _legacy_80_20()
        return 0
    if not all(
        [args.train_csv, args.val_csv, args.test_csv, args.images_dir, args.output_dir]
    ):
        print(
            "Error: provide --train-csv, --val-csv, --test-csv, --images-dir, and --output-dir, "
            "or use --legacy-80-20 for the previous single-CSV random split.",
            file=sys.stderr,
        )
        return 2
    _train_experiment_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
