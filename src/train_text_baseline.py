#!/usr/bin/env python3
"""TF–IDF + linear classifier for OCR text (same split files as the CNN; hybrid can reuse splits)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import LABELS, NUM_CLASSES  # noqa: E402
from experiment_prep import (  # noqa: E402
    active_original_labels,
    apply_exclude,
    build_label_mappings,
    downsample_train_label,
    load_split_dataframe,
    remap_labels_column,
)
from layout_features import (  # noqa: E402
    LINE_NO_COLUMN,
    active_layout_columns,
    validate_layout_dataframe_columns,
)
from metrics_reporting import (  # noqa: E402
    compute_split_metrics,
    confusion_matrix_to_csv,
    save_json,
)


def _df_to_xy(d: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    if "text" not in d.columns or "label" not in d.columns:
        raise ValueError("DataFrame must have columns: text, label")
    if d["label"].isna().any():
        raise ValueError("Empty labels")
    dc = d.copy()
    dc["text"] = dc["text"].fillna("").astype(str)
    return dc["text"].tolist(), dc["label"].values.astype(int)


def _prepare_text_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["text"] = out["text"].fillna("").astype(str)
    return out


def _prepare_layout_frame(d: pd.DataFrame, layout_cols: List[str]) -> pd.DataFrame:
    validate_layout_dataframe_columns(d.columns, exclude_line_no=LINE_NO_COLUMN not in layout_cols)
    out = _prepare_text_frame(d)
    for col in layout_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")
    return out


def _model_key(
    model: str,
    use_layout: bool,
    exclude_line_no: bool,
) -> str:
    base = "text_logreg" if model == "logistic_regression" else "text_svm"
    if not use_layout:
        return base
    if exclude_line_no:
        return f"{base}_layout_no_line_no"
    return f"{base}_layout"


def _build_pipeline(
    model: str,
    ngram_min: int,
    ngram_max: int,
    max_features: Optional[int],
    seed: int,
    class_weight: Optional[str],
    use_layout: bool = False,
    layout_cols: Optional[List[str]] = None,
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

    if not use_layout:
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_kw)),
                ("clf", clf),
            ]
        )

    cols = layout_cols or active_layout_columns(exclude_line_no=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(**tfidf_kw), "text"),
            ("layout", StandardScaler(with_mean=False), cols),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def _fit_predict(
    pipe: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_layout: bool,
    layout_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if use_layout:
        x_tr = _prepare_layout_frame(train_df, layout_cols)
        x_va = _prepare_layout_frame(val_df, layout_cols)
        x_te = _prepare_layout_frame(test_df, layout_cols)
        y_tr = train_df["label"].values.astype(int)
        pipe.fit(x_tr, y_tr)
        return pipe.predict(x_va), pipe.predict(x_te)
    texts_tr, y_tr = _df_to_xy(train_df)
    texts_va, _ = _df_to_xy(val_df)
    texts_te, _ = _df_to_xy(test_df)
    pipe.fit(texts_tr, y_tr)
    return pipe.predict(texts_va), pipe.predict(texts_te)


def _to_original_labels(y: np.ndarray, train_to_orig: Dict[int, int]) -> np.ndarray:
    y = np.asarray(y).ravel().astype(int)
    return np.array([train_to_orig[int(i)] for i in y], dtype=int)


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
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=("balanced", "none"),
    )
    p.add_argument(
        "--exclude-labels",
        type=int,
        nargs="*",
        default=None,
        help="Original label ids to drop (e.g. 10 for CURRENCY).",
    )
    p.add_argument("--downsample-label", type=int, default=None)
    p.add_argument("--downsample-ratio", type=float, default=None)
    p.add_argument(
        "--use-layout-features",
        action="store_true",
        help="Use bbox/layout numeric features together with TF-IDF text.",
    )
    p.add_argument(
        "--exclude-line-no-feature",
        action="store_true",
        help="When --use-layout-features is set, omit line_no_norm from layout features.",
    )
    args = p.parse_args()

    if args.exclude_line_no_feature and not args.use_layout_features:
        print(
            "Error: --exclude-line-no-feature requires --use-layout-features.",
            file=sys.stderr,
        )
        return 2

    if (args.downsample_label is not None) ^ (args.downsample_ratio is not None):
        print(
            "Error: provide both --downsample-label and --downsample-ratio, or neither.",
            file=sys.stderr,
        )
        return 2

    for path in (args.train_csv, args.val_csv, args.test_csv):
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            return 1

    excluded = set(int(x) for x in (args.exclude_labels or []))
    for lid in excluded:
        if lid < 0 or lid >= NUM_CLASSES:
            print(f"exclude label {lid} out of range 0..{NUM_CLASSES - 1}", file=sys.stderr)
            return 2
    if args.downsample_label is not None and int(args.downsample_label) in excluded:
        print(
            "Error: cannot downsample a label that is also excluded.",
            file=sys.stderr,
        )
        return 2

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
        print("Error: train, val, and test must be non-empty after exclusions.", file=sys.stderr)
        return 2

    down_summary: Dict[str, Any] = {"applied": False}
    if args.downsample_label is not None:
        train_df, down_summary = downsample_train_label(
            train_df,
            int(args.downsample_label),
            float(args.downsample_ratio),
            int(args.seed),
        )
        print(
            f"Train downsampling: label={args.downsample_label}, ratio={args.downsample_ratio}, "
            f"applied={down_summary.get('applied')}, rows_removed={down_summary.get('rows_removed', 0)}"
        )
        if len(train_df) == 0:
            print("Error: training set empty after downsampling.", file=sys.stderr)
            return 2

    active_orig = active_original_labels(NUM_CLASSES, excluded)
    if not active_orig:
        print("Error: no active labels.", file=sys.stderr)
        return 2
    orig_to_train, train_to_orig = build_label_mappings(active_orig)
    num_active = len(active_orig)

    train_df_m = remap_labels_column(train_df, orig_to_train)
    val_df_m = remap_labels_column(val_df, orig_to_train)
    test_df_m = remap_labels_column(test_df, orig_to_train)

    layout_cols = active_layout_columns(exclude_line_no=args.exclude_line_no_feature)
    line_no_used = args.use_layout_features and not args.exclude_line_no_feature

    cw = "balanced" if args.class_weight == "balanced" else None
    mf = args.max_features
    if mf is not None and mf <= 0:
        mf = None

    y_va = val_df_m["label"].values.astype(int)
    y_te = test_df_m["label"].values.astype(int)

    pipe = _build_pipeline(
        args.model,
        int(args.ngram_min),
        int(args.ngram_max),
        mf,
        int(args.seed),
        cw,
        use_layout=args.use_layout_features,
        layout_cols=layout_cols if args.use_layout_features else None,
    )
    y_val_pred, y_test_pred = _fit_predict(
        pipe,
        train_df_m,
        val_df_m,
        test_df_m,
        args.use_layout_features,
        layout_cols,
    )

    y_va_o = _to_original_labels(y_va, train_to_orig)
    y_te_o = _to_original_labels(y_te, train_to_orig)
    y_val_po = _to_original_labels(y_val_pred, train_to_orig)
    y_test_po = _to_original_labels(y_test_pred, train_to_orig)

    report_labels = active_orig
    val_m = compute_split_metrics(
        y_va_o, y_val_po, num_classes=num_active, labels=report_labels
    )
    test_m = compute_split_metrics(
        y_te_o, y_test_po, num_classes=num_active, labels=report_labels
    )
    val_cm = confusion_matrix(
        y_va_o, y_val_po, labels=np.array(report_labels)
    )
    test_cm = confusion_matrix(
        y_te_o, y_test_po, labels=np.array(report_labels)
    )
    val_rep = classification_report(
        y_va_o,
        y_val_po,
        labels=report_labels,
        target_names=[LABELS[i] for i in report_labels],
        output_dict=True,
        zero_division=0,
    )
    test_rep = classification_report(
        y_te_o,
        y_test_po,
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
    if args.use_layout_features:
        model_name += "_Layout"
        if args.exclude_line_no_feature:
            model_name += "_NoLineNo"

    input_type = "text_ocr+layout" if args.use_layout_features else "text_ocr"
    model_key = _model_key(
        args.model,
        args.use_layout_features,
        args.exclude_line_no_feature,
    )

    config_payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_key": model_key,
        "input_type": input_type,
        "layout_features_used": args.use_layout_features,
        "line_no_feature_used": line_no_used,
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
    if args.use_layout_features:
        config_payload["layout_feature_columns"] = layout_cols

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
            "model_name": model_name,
            "model_key": model_key,
            "input_type": input_type,
            "layout_features_used": args.use_layout_features,
            "line_no_feature_used": line_no_used,
            "val": val_m,
            "test": test_m,
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
    joblib.dump(pipe, out / "model.joblib")
    print("=== Val ===", val_m)
    print("=== Test ===", test_m)
    print(f"Saved {out / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
