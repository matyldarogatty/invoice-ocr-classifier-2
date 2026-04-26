"""Shared metrics computation and export for CNN and text experiments (hybrid-friendly)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if labels is None:
        labels = list(range(num_classes))
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    )
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class: Dict[str, Dict[str, float]] = {}
    for k, v in rep.items():
        if k in ("accuracy", "macro avg", "weighted avg") or not isinstance(v, dict):
            continue
        if "precision" in v:
            per_class[k] = {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1-score": float(v["f1-score"]),
                "support": int(v["support"]),
            }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def confusion_matrix_to_csv(
    path: Path,
    cm: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    label_order: Optional[List[int]] = None,
) -> None:
    """
    If label_order is set, row/column i corresponds to original label id label_order[i]
    (for sklearn confusion_matrix(..., labels=label_order)).
    Otherwise row/column i is treated as label id i (backward compatible).
    """
    n = cm.shape[0]
    names = label_names or {}
    col_names: List[str] = []
    row_names: List[str] = []
    for i in range(n):
        oid = label_order[i] if label_order is not None else i
        name = names.get(oid, str(oid))
        col_names.append(f"pred_{name}")
        row_names.append(f"true_{name}")
    df = pd.DataFrame(cm, index=row_names, columns=col_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
