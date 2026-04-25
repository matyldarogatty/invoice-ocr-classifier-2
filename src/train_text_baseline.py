#!/usr/bin/env python3
"""TF–IDF + linear classifier for OCR text (same split files as the CNN; hybrid can reuse splits)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import LABELS, NUM_CLASSES  # noqa: E402
from metrics_reporting import (  # noqa: E402
    compute_split_metrics,
    confusion_matrix_to_csv,
    save_json,
)


def _load_xy(path: Path) -> Tuple[List[str], np.ndarray]:
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    if "text" not in d.columns or "label" not in d.columns:
        raise ValueError(f"{path} must have columns: text, label")
    if d["label"].isna().any():
        raise ValueError(f"Empty labels in {path}")
    d = d.copy()
    d["text"] = d["text"].fillna("").astype(str)
    y = d["label"].values.astype(int)
    texts = d["text"].tolist()
    return texts, y


def _build_pipeline(
    model: str,
    ngram_min: int,
    ngram_max: int,
    max_features: Optional[int],
    seed: int,
    class_weight: Optional[str],
) -> Pipeline:
    tfidf_kw: Dict[str, Any] = {
        "ngram_range": (ngram_min, ngram_max),
        "sublinear_tf": True,
    }
    if max_features is not None and max_features > 0:
        tfidf_kw["max_features"] = int(max_features)
    if model == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=seed,
        )
    elif model == "linear_svc":
        clf = LinearSVC(
            class_weight=class_weight,
            max_iter=10000,
            dual="auto",
            random_state=seed,
        )
    else:
        raise ValueError("model must be logistic_regression or linear_svc")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_kw)),
            ("clf", clf),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description="TF-IDF + linear classifier for line text")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=("logistic_regression", "linear_svc"),
    )
    p.add_argument("--max-features", type=int, default=None, help="TfidfVectorizer max_features (default: unbounded)")
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=("balanced", "none"),
        help="sklearn class_weight: balanced or None (no reweighting).",
    )
    args = p.parse_args()

    for path in (args.train_csv, args.val_csv, args.test_csv):
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            return 1

    cw = "balanced" if args.class_weight == "balanced" else None
    mf = args.max_features
    if mf is not None and mf <= 0:
        mf = None
    texts_tr, y_tr = _load_xy(args.train_csv)
    texts_va, y_va = _load_xy(args.val_csv)
    texts_te, y_te = _load_xy(args.test_csv)
    if len(texts_tr) == 0 or len(texts_va) == 0 or len(texts_te) == 0:
        print("Error: train, val, and test must be non-empty.", file=sys.stderr)
        return 2

    pipe = _build_pipeline(
        args.model,
        int(args.ngram_min),
        int(args.ngram_max),
        mf,
        int(args.seed),
        cw,
    )
    pipe.fit(texts_tr, y_tr)
    y_val_pred = pipe.predict(texts_va)
    y_test_pred = pipe.predict(texts_te)
    report_labels = list(range(NUM_CLASSES))
    val_m = compute_split_metrics(
        y_va, y_val_pred, num_classes=NUM_CLASSES, labels=report_labels
    )
    test_m = compute_split_metrics(
        y_te, y_test_pred, num_classes=NUM_CLASSES, labels=report_labels
    )
    val_cm = confusion_matrix(
        y_va, y_val_pred, labels=np.arange(NUM_CLASSES)
    )
    test_cm = confusion_matrix(
        y_te, y_test_pred, labels=np.arange(NUM_CLASSES)
    )
    val_rep = classification_report(
        y_va,
        y_val_pred,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )
    test_rep = classification_report(
        y_te,
        y_test_pred,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = (
        f"Tfidf_{'LogReg' if args.model == 'logistic_regression' else 'LinearSVC'}"
    )
    config_payload: Dict[str, Any] = {
        "model_name": model_name,
        "input_type": "text_ocr",
        "train_csv": str(args.train_csv.resolve()),
        "val_csv": str(args.val_csv.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "sklearn_model": args.model,
        "ngram_min": int(args.ngram_min),
        "ngram_max": int(args.ngram_max),
        "max_features": mf,
        "seed": int(args.seed),
        "class_weight": args.class_weight,
        "missing_text_policy": "replace_with_empty_string",
        "dropped_rows_train": 0,
        "dropped_rows_val": 0,
        "dropped_rows_test": 0,
    }
    save_json(out / "config.json", config_payload)
    save_json(
        out / "metrics.json",
        {
            "model_name": model_name,
            "input_type": "text_ocr",
            "val": val_m,
            "test": test_m,
        },
    )
    save_json(out / "classification_report_val.json", val_rep)
    save_json(out / "classification_report_test.json", test_rep)
    confusion_matrix_to_csv(out / "confusion_matrix_val.csv", val_cm, label_names=LABELS)
    confusion_matrix_to_csv(
        out / "confusion_matrix_test.csv", test_cm, label_names=LABELS
    )
    joblib.dump(pipe, out / "model.joblib")
    print("=== Val ===", val_m)
    print("=== Test ===", test_m)
    print(f"Saved {out / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
