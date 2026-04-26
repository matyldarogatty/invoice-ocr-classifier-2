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
from experiment_prep import (
    active_original_labels,
    apply_exclude,
    build_label_mappings,
    downsample_train_label,
    load_split_dataframe,
    remap_labels_column,
)
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


def _check_images_from_frames(
    frames: List[pd.DataFrame], images_dir: Path, limit_msg: int = 5
) -> None:
    missing: List[str] = []
    for d in frames:
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
    y_train: np.ndarray, num_active: int, use_weights: bool, device: str
) -> Optional[torch.Tensor]:
    if not use_weights:
        return None
    y = np.asarray(y_train).astype(int).ravel()
    unique = np.unique(y)
    cw = sk_class_weight.compute_class_weight("balanced", classes=unique, y=y)
    w = np.ones(num_active, dtype=np.float32)
    for c, wgt in zip(unique, cw):
        w[int(c)] = wgt
    return torch.from_numpy(w).to(device)


def _to_original_labels(y: np.ndarray, train_to_orig: Dict[int, int]) -> np.ndarray:
    y = np.asarray(y).ravel().astype(int)
    return np.array([train_to_orig[int(i)] for i in y], dtype=int)


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

    excluded = set(int(x) for x in (args.exclude_labels or []))
    for lid in excluded:
        if lid < 0 or lid >= NUM_CLASSES:
            raise ValueError(
                f"exclude label {lid} out of range 0..{NUM_CLASSES - 1}"
            )

    train_df = load_split_dataframe(args.train_csv)
    val_df = load_split_dataframe(args.val_csv)
    test_df = load_split_dataframe(args.test_csv)
    n_tr0, n_va0, n_te0 = len(train_df), len(val_df), len(test_df)

    train_df, r_tr_e = apply_exclude(train_df, excluded)
    val_df, r_va_e = apply_exclude(val_df, excluded)
    test_df, r_te_e = apply_exclude(test_df, excluded)
    if excluded:
        print(
            f"Excluded labels {sorted(excluded)}: removed rows — "
            f"train {r_tr_e}/{n_tr0}, val {r_va_e}/{n_va0}, test {r_te_e}/{n_te0}"
        )

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("After exclusions, train/val/test must be non-empty.")

    down_summary: Dict[str, Any] = {"applied": False}
    if getattr(args, "downsample_label", None) is not None:
        dl = int(args.downsample_label)
        if dl in excluded:
            raise ValueError("Cannot --downsample-label a class that is also excluded.")
        dr = float(args.downsample_ratio)
        train_df, down_summary = downsample_train_label(
            train_df, dl, dr, int(args.seed)
        )
        print(
            f"Train downsampling: label={dl}, ratio={dr}, "
            f"applied={down_summary.get('applied')}, rows_removed={down_summary.get('rows_removed', 0)}"
        )
        if len(train_df) == 0:
            raise ValueError("Training set empty after downsampling.")

    _check_images_from_frames(
        [train_df, val_df, test_df],
        Path(args.images_dir),
    )

    active_orig = active_original_labels(NUM_CLASSES, excluded)
    if not active_orig:
        raise ValueError("No active labels after exclusions.")
    orig_to_train, train_to_orig = build_label_mappings(active_orig)
    num_active = len(active_orig)

    train_df_m = remap_labels_column(train_df, orig_to_train)
    val_df_m = remap_labels_column(val_df, orig_to_train)
    test_df_m = remap_labels_column(test_df, orig_to_train)

    _set_seed(int(args.seed))
    device = _device_from_arg(str(args.device))

    train_set = InvoiceDataset(
        csv_path=None,
        images_dir=str(args.images_dir),
        dataframe=train_df_m,
        strict=True,
    )
    val_set = InvoiceDataset(
        csv_path=None,
        images_dir=str(args.images_dir),
        dataframe=val_df_m,
        strict=True,
    )
    test_set = InvoiceDataset(
        csv_path=None,
        images_dir=str(args.images_dir),
        dataframe=test_df_m,
        strict=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=int(args.batch_size), shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=int(args.batch_size), shuffle=False
    )

    model = InvoiceCNN(num_classes=num_active).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate)
    )
    y_tr = train_df_m["label"].values.astype(int)
    ce_weight = _build_class_weights(
        y_tr, num_active, bool(args.use_class_weights), device
    )
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    report_labels = active_orig

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

    y_val_t_o = _to_original_labels(y_val_t, train_to_orig)
    y_val_p_o = _to_original_labels(y_val_p, train_to_orig)
    y_test_t_o = _to_original_labels(y_test_t, train_to_orig)
    y_test_p_o = _to_original_labels(y_test_p, train_to_orig)

    val_metrics = compute_split_metrics(
        y_val_t_o, y_val_p_o, num_classes=num_active, labels=report_labels
    )
    test_metrics = compute_split_metrics(
        y_test_t_o, y_test_p_o, num_classes=num_active, labels=report_labels
    )
    val_cm = confusion_matrix(
        y_val_t_o, y_val_p_o, labels=np.array(report_labels)
    )
    test_cm = confusion_matrix(
        y_test_t_o, y_test_p_o, labels=np.array(report_labels)
    )
    val_rep = classification_report(
        y_val_t_o,
        y_val_p_o,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )
    test_rep = classification_report(
        y_test_t_o,
        y_test_p_o,
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
        "config_num_classes": NUM_CLASSES,
        "active_original_labels": active_orig,
        "excluded_label_ids": sorted(excluded),
        "model_output_classes": num_active,
        "rows_removed_by_exclude": {
            "train": r_tr_e,
            "val": r_va_e,
            "test": r_te_e,
        },
        "train_downsampling": down_summary,
    }
    save_json(out / "config.json", config_payload)
    save_json(
        out / "label_mapping.json",
        {
            "active_original_labels": active_orig,
            "excluded_labels": sorted(excluded),
            "original_to_training": {str(k): v for k, v in orig_to_train.items()},
            "training_to_original": {str(k): v for k, v in train_to_orig.items()},
            "config_num_classes": NUM_CLASSES,
        },
    )
    if down_summary.get("applied"):
        save_json(out / "sampling_summary.json", down_summary)
    save_json(
        out / "metrics.json",
        {
            "model_name": "InvoiceCNN",
            "input_type": "image",
            "val": val_metrics,
            "test": test_metrics,
            "metrics_label_space": "original_ids_active_only",
            "active_original_labels": active_orig,
        },
    )
    save_json(out / "classification_report_val.json", val_rep)
    save_json(out / "classification_report_test.json", test_rep)
    confusion_matrix_to_csv(
        out / "confusion_matrix_val.csv", val_cm, label_names=LABELS, label_order=report_labels
    )
    confusion_matrix_to_csv(
        out / "confusion_matrix_test.csv",
        test_cm,
        label_names=LABELS,
        label_order=report_labels,
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
    p.add_argument(
        "--exclude-labels",
        type=int,
        nargs="*",
        default=None,
        help="Original label ids to drop from train/val/test in memory (e.g. 10 for CURRENCY).",
    )
    p.add_argument(
        "--downsample-label",
        type=int,
        default=None,
        help="Train only: cap this original label's count to ratio × max count of other labels.",
    )
    p.add_argument(
        "--downsample-ratio",
        type=float,
        default=None,
        help="Must be used with --downsample-label (e.g. 3.0).",
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
    if (args.downsample_label is not None) ^ (args.downsample_ratio is not None):
        print(
            "Error: provide both --downsample-label and --downsample-ratio, or neither.",
            file=sys.stderr,
        )
        return 2
    _train_experiment_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
