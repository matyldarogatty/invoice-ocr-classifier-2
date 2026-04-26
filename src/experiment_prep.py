"""Shared experiment prep: exclude labels, remap for training, train-only downsampling (no CSV writes)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def active_original_labels(num_classes: int, excluded: Set[int]) -> List[int]:
    return [i for i in range(num_classes) if i not in excluded]


def build_label_mappings(active_orig: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    orig_to_train = {o: i for i, o in enumerate(active_orig)}
    train_to_orig = {i: o for o, i in orig_to_train.items()}
    return orig_to_train, train_to_orig


def apply_exclude(df: pd.DataFrame, excluded: Set[int]) -> Tuple[pd.DataFrame, int]:
    if not excluded:
        return df.reset_index(drop=True), 0
    mask = ~df["label"].isin(excluded)
    removed = int((~mask).sum())
    return df.loc[mask].reset_index(drop=True), removed


def remap_labels_column(df: pd.DataFrame, orig_to_train: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].map(orig_to_train).astype(int)
    return out


def downsample_train_label(
    df: pd.DataFrame,
    label_id: int,
    ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cap rows with label_id to at most ratio * max(count of other labels). Train-only caller."""
    if ratio <= 0:
        raise ValueError("downsample_ratio must be positive")
    counts = df["label"].value_counts().to_dict()
    if label_id not in counts:
        return df.reset_index(drop=True), {
            "applied": False,
            "reason": f"label {label_id} not in training set",
            "train_class_counts_before": {str(k): int(v) for k, v in sorted(counts.items())},
            "train_class_counts_after": {str(k): int(v) for k, v in sorted(counts.items())},
        }
    other_counts = [c for lid, c in counts.items() if lid != label_id]
    max_other = max(other_counts) if other_counts else 0
    cap = int(ratio * max_other)
    n_cur = int(counts[label_id])
    before = {str(k): int(v) for k, v in sorted(counts.items())}
    if max_other == 0:
        return df.reset_index(drop=True), {
            "applied": False,
            "reason": "no non-downsampled classes with rows; skipping",
            "train_class_counts_before": before,
            "train_class_counts_after": before,
            "cap_computed": cap,
        }
    if n_cur <= cap:
        return df.reset_index(drop=True), {
            "applied": False,
            "reason": "already at or below cap",
            "train_class_counts_before": before,
            "train_class_counts_after": before,
            "cap": cap,
            "max_non_downsampled_count": max_other,
        }
    rng = random.Random(seed)
    idx_all = df.index[df["label"] == label_id].tolist()
    keep_idx = set(rng.sample(idx_all, cap))
    drop_idx = [i for i in idx_all if i not in keep_idx]
    out = df.drop(index=drop_idx).reset_index(drop=True)
    after_counts = out["label"].value_counts().to_dict()
    after = {str(k): int(v) for k, v in sorted(after_counts.items())}
    return out, {
        "applied": True,
        "downsample_label": label_id,
        "downsample_ratio": ratio,
        "max_non_downsampled_count": max_other,
        "cap": cap,
        "rows_removed": len(drop_idx),
        "train_class_counts_before": before,
        "train_class_counts_after": after,
    }


def load_split_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    d = pd.read_csv(p)
    d.columns = [str(c).strip() for c in d.columns]
    return d
