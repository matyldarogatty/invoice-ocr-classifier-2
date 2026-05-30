#!/usr/bin/env python3
"""Run document splits + CNN/text training for multiple seeds; aggregate metrics for thesis."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC = PROJECT_DIR / "src"


def _run(cmd: List[str], cwd: Path) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(seed: int, model_key: str, exp_dir: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    def g(split: str, key: str) -> float | None:
        v = data.get(split, {}).get(key)
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "seed": seed,
        "model_key": model_key,
        "model_name": data.get("model_name", model_key),
        "input_type": data.get("input_type", ""),
        "layout_features_used": data.get("layout_features_used", False),
        "line_no_feature_used": data.get("line_no_feature_used", False),
        "experiment_dir": str(exp_dir.resolve()),
        "val_accuracy": g("val", "accuracy"),
        "val_macro_f1": g("val", "macro_f1"),
        "val_weighted_f1": g("val", "weighted_f1"),
        "test_accuracy": g("test", "accuracy"),
        "test_macro_f1": g("test", "macro_f1"),
        "test_weighted_f1": g("test", "weighted_f1"),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)

    summary: List[Dict[str, Any]] = []
    metric_cols = [
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    for model_key, group in sorted(by_model.items()):
        entry: Dict[str, Any] = {
            "model_key": model_key,
            "model_name": group[0]["model_name"],
            "input_type": group[0]["input_type"],
            "layout_features_used": group[0].get("layout_features_used", False),
            "line_no_feature_used": group[0].get("line_no_feature_used", False),
            "n_seeds": len(group),
            "seeds": sorted(r["seed"] for r in group),
        }
        for col in metric_cols:
            vals = [r[col] for r in group if r[col] is not None]
            if vals:
                entry[f"{col}_mean"] = round(mean(vals), 4)
                entry[f"{col}_std"] = round(stdev(vals), 4) if len(vals) > 1 else 0.0
        summary.append(entry)
    return summary


def _collect_per_class_rows(
    seed: int,
    model_key: str,
    exp_dir: Path,
) -> List[Dict[str, Any]]:
    report_path = exp_dir / "classification_report_test.json"
    if not report_path.is_file():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for class_name, metrics in report.items():
        if class_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(metrics, dict) or "f1-score" not in metrics:
            continue
        rows.append(
            {
                "seed": seed,
                "model_key": model_key,
                "class_name": class_name,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1-score", 0.0)),
                "support": int(metrics.get("support", 0)),
                "experiment_dir": str(exp_dir.resolve()),
            }
        )
    return rows


def _run_cnn(
    *,
    seed: int,
    model_key: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    images_dir: Path,
    output_root: Path,
    exclude_labels: List[int],
    epochs: int,
    batch_size: int,
    device: str,
    use_layout: bool,
    exclude_line_no: bool,
) -> Path:
    cnn_dir = output_root / f"exp_{model_key}_seed{seed}"
    cmd = [
        sys.executable,
        str(SRC / "train.py"),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--test-csv",
        str(test_csv),
        "--images-dir",
        str(images_dir),
        "--output-dir",
        str(cnn_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--seed",
        str(seed),
        "--use-class-weights",
    ]
    if exclude_labels:
        cmd.extend(["--exclude-labels", *map(str, exclude_labels)])
    if use_layout:
        cmd.append("--use-layout-features")
    if exclude_line_no:
        cmd.append("--exclude-line-no-feature")
    _run(cmd, PROJECT_DIR)
    return cnn_dir


def _run_text_baseline(
    *,
    seed: int,
    model_key: str,
    model_arg: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_root: Path,
    exclude_labels: List[int],
    use_layout: bool,
    exclude_line_no: bool,
) -> Path:
    text_dir = output_root / f"exp_{model_key}_seed{seed}"
    cmd = [
        sys.executable,
        str(SRC / "train_text_baseline.py"),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--test-csv",
        str(test_csv),
        "--output-dir",
        str(text_dir),
        "--seed",
        str(seed),
        "--model",
        model_arg,
    ]
    if exclude_labels:
        cmd.extend(["--exclude-labels", *map(str, exclude_labels)])
    if use_layout:
        cmd.append("--use-layout-features")
    if exclude_line_no:
        cmd.append("--exclude-line-no-feature")
    _run(cmd, PROJECT_DIR)
    return text_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-seed experiment runner for thesis metrics.")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for document splits and training.",
    )
    p.add_argument("--labels-csv", type=Path, default=PROJECT_DIR / "data" / "labels_synthetic.csv")
    p.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "data" / "images_synthetic")
    p.add_argument("--output-root", type=Path, default=PROJECT_DIR / "output" / "multi_seed")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--exclude-labels", type=int, nargs="*", default=[10])
    p.add_argument("--skip-cnn", action="store_true")
    p.add_argument("--skip-text", action="store_true")
    p.add_argument(
        "--with-layout",
        action="store_true",
        help="Run layout text experiments (uses layout labels CSV by default).",
    )
    p.add_argument(
        "--layout-labels-csv",
        type=Path,
        default=PROJECT_DIR / "data" / "labels_synthetic_with_layout.csv",
        help="Labels CSV with layout feature columns (used when --with-layout or --with-cnn-layout).",
    )
    p.add_argument(
        "--with-cnn-layout",
        action="store_true",
        help="Run CNN + layout experiments (cnn, cnn_layout, cnn_layout_no_line_no).",
    )
    args = p.parse_args()

    if args.with_layout and args.with_cnn_layout:
        print("Error: use --with-layout or --with-cnn-layout, not both.", file=sys.stderr)
        return 2

    if args.with_cnn_layout:
        labels_csv = args.layout_labels_csv
        if args.output_root == PROJECT_DIR / "output" / "multi_seed":
            args.output_root = PROJECT_DIR / "output" / "cnn_layout_experiments_450"
    elif args.with_layout:
        labels_csv = args.layout_labels_csv
        if args.output_root == PROJECT_DIR / "output" / "multi_seed":
            args.output_root = PROJECT_DIR / "output" / "layout_experiments"
        args.skip_cnn = True
    else:
        labels_csv = args.labels_csv

    if not labels_csv.is_file():
        print(f"Labels CSV not found: {labels_csv}", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        split_dir = args.output_root / f"splits_seed{seed}"
        _run(
            [
                sys.executable,
                str(SRC / "create_splits.py"),
                "--labels-csv",
                str(labels_csv),
                "--output-dir",
                str(split_dir),
                "--seed",
                str(seed),
            ],
            PROJECT_DIR,
        )

        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"

        if args.with_cnn_layout and not args.skip_cnn:
            cnn_variants: List[tuple[str, bool, bool]] = [
                ("cnn", False, False),
                ("cnn_layout", True, False),
                ("cnn_layout_no_line_no", True, True),
            ]
            for model_key, use_layout, exclude_line_no in cnn_variants:
                cnn_dir = _run_cnn(
                    seed=seed,
                    model_key=model_key,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    images_dir=args.images_dir,
                    output_root=args.output_root,
                    exclude_labels=list(args.exclude_labels),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    use_layout=use_layout,
                    exclude_line_no=exclude_line_no,
                )
                metrics = _load_metrics(cnn_dir / "metrics.json")
                all_rows.append(_metric_row(seed, model_key, cnn_dir, metrics))
                per_class_rows.extend(_collect_per_class_rows(seed, model_key, cnn_dir))
        elif not args.skip_cnn and not args.with_layout:
            cnn_dir = args.output_root / f"exp_cnn_seed{seed}"
            cmd = [
                sys.executable,
                str(SRC / "train.py"),
                "--train-csv",
                str(train_csv),
                "--val-csv",
                str(val_csv),
                "--test-csv",
                str(test_csv),
                "--images-dir",
                str(args.images_dir),
                "--output-dir",
                str(cnn_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--use-class-weights",
            ]
            if args.exclude_labels:
                cmd.extend(["--exclude-labels", *map(str, args.exclude_labels)])
            _run(cmd, PROJECT_DIR)
            all_rows.append(_metric_row(seed, "cnn", cnn_dir, _load_metrics(cnn_dir / "metrics.json")))
            per_class_rows.extend(_collect_per_class_rows(seed, "cnn", cnn_dir))

        if not args.skip_text:
            text_variants: List[tuple[str, str, bool, bool]] = [
                ("text_logreg", "logistic_regression", False, False),
                ("text_svm", "linear_svc", False, False),
            ]
            if args.with_layout:
                text_variants.extend(
                    [
                        ("text_logreg_layout", "logistic_regression", True, False),
                        ("text_svm_layout", "linear_svc", True, False),
                        ("text_logreg_layout_no_line_no", "logistic_regression", True, True),
                        ("text_svm_layout_no_line_no", "linear_svc", True, True),
                    ]
                )

            for model_key, model_arg, use_layout, exclude_line_no in text_variants:
                text_dir = _run_text_baseline(
                    seed=seed,
                    model_key=model_key,
                    model_arg=model_arg,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    output_root=args.output_root,
                    exclude_labels=list(args.exclude_labels),
                    use_layout=use_layout,
                    exclude_line_no=exclude_line_no,
                )
                metrics = _load_metrics(text_dir / "metrics.json")
                all_rows.append(_metric_row(seed, model_key, text_dir, metrics))
                per_class_rows.extend(_collect_per_class_rows(seed, model_key, text_dir))

    args.output_root.mkdir(parents=True, exist_ok=True)
    per_run_csv = args.output_root / "multi_seed_per_run.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        w.writeheader()
        w.writerows(all_rows)

    summary = _aggregate(all_rows)
    summary_json = args.output_root / "multi_seed_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = args.output_root / "multi_seed_summary.csv"
    if summary:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    per_class_csv = args.output_root / "per_class_f1.csv"
    if per_class_rows:
        with per_class_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
            w.writeheader()
            w.writerows(per_class_rows)

    print("\n=== Multi-seed summary (test_macro_f1 mean) ===")
    for s in summary:
        m = s.get("test_macro_f1_mean", "n/a")
        print(f"  {s['model_key']:32}  test_macro_f1_mean={m}")
    print(f"Wrote {per_run_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    if per_class_rows:
        print(f"Wrote {per_class_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
