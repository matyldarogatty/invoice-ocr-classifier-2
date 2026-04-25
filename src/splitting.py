"""Document-level dataset splitting (prepare for future hybrid: same group ids for image + text)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resolve_group_column(
    df: pd.DataFrame, group_column: Optional[str] = None
) -> str:
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(
                f"--group-column {group_column!r} not found in CSV columns: {list(df.columns)}"
            )
        if df[group_column].isna().all():
            raise ValueError(
                f"Group column {group_column!r} is all null; cannot split at document level."
            )
        return group_column
    if "invoice_id" in df.columns and not df["invoice_id"].isna().all():
        return "invoice_id"
    if "source_pdf" in df.columns and not df["source_pdf"].isna().all():
        return "source_pdf"
    raise ValueError(
        "Document-level splitting is impossible: no 'invoice_id' or 'source_pdf' column with "
        "at least one non-null value, and no --group-column was provided. "
        "Add an invoice or document id column, or pass --group-column with a valid column name."
    )


def _split_sizes(n: int, train_r: float, val_r: float, test_r: float) -> Tuple[int, int, int]:
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_r + val_r + test_r}")
    if n < 1:
        raise ValueError("No document groups to split.")
    if n < 3:
        raise ValueError(
            f"Need at least 3 document groups for train/val/test (got n={n}). "
            "Add more documents or use a larger dataset."
        )
    # Largest-remainder method: integer counts sum to n
    exact = np.array([train_r, val_r, test_r], dtype=np.float64) * n
    floors = np.floor(exact).astype(int)
    frac = exact - floors
    rem = n - int(floors.sum())
    idx = np.argsort(-frac)
    for k in range(rem):
        floors[idx[k]] += 1
    # When n is small, LRM can assign 0 to a split; ensure each part has at least 1 for n >= 3
    while n >= 3 and (floors.min() < 1) and (floors.max() > 1):
        j = int(np.argmin(floors))
        k = int(np.argmax(floors))
        floors[k] -= 1
        floors[j] += 1
    n_train, n_val, n_test = int(floors[0]), int(floors[1]), int(floors[2])
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Empty split: train={n_train}, val={n_val}, test={n_test} for n_docs={n} "
            "(add more documents or adjust ratios)."
        )
    if n_train + n_val + n_test != n:
        raise RuntimeError("Internal error: document counts do not sum to n_docs")
    return n_train, n_val, n_test


def document_level_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    gdf = df.copy()
    if group_col not in gdf.columns:
        raise ValueError(f"Missing group column: {group_col!r}")
    if gdf[group_col].isna().any():
        n = int(gdf[group_col].isna().sum())
        raise ValueError(
            f"Group column {group_col!r} has {n} null row(s). "
            "Fill or remove them before document-level splitting."
        )
    gdf["_doc_key"] = gdf[group_col].astype(str)
    if (gdf["_doc_key"].str.strip() == "").any():
        raise ValueError(
            f"Group column {group_col!r} has empty string(s); use a non-empty document identifier per row."
        )
    group_ids: List[str] = sorted(gdf["_doc_key"].unique().tolist())
    n = len(group_ids)
    n_train, n_val, n_test = _split_sizes(n, train_ratio, val_ratio, test_ratio)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(group_ids)
    train_ids = set(perm[:n_train].tolist())
    val_ids = set(perm[n_train : n_train + n_val].tolist())
    test_ids = set(perm[n_train + n_val :].tolist())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Internal error: document ids overlap between splits")
    tr = gdf[gdf["_doc_key"].isin(train_ids)].drop(columns=["_doc_key"])
    va = gdf[gdf["_doc_key"].isin(val_ids)].drop(columns=["_doc_key"])
    te = gdf[gdf["_doc_key"].isin(test_ids)].drop(columns=["_doc_key"])
    meta: Dict[str, Any] = {
        "group_column": group_col,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_documents_train": n_train,
        "n_documents_val": n_val,
        "n_documents_test": n_test,
        "n_rows_train": len(tr),
        "n_rows_val": len(va),
        "n_rows_test": len(te),
    }
    return tr, va, te, meta


def class_distribution_str(df: pd.DataFrame) -> Dict[str, int]:
    if "label" not in df.columns:
        return {}
    c = df["label"].value_counts()
    return {str(k): int(v) for k, v in c.items()}


def build_split_metadata(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        **base,
        "class_distribution_train": class_distribution_str(train),
        "class_distribution_val": class_distribution_str(val),
        "class_distribution_test": class_distribution_str(test),
    }
