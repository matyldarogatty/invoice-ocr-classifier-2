# Experiment pipeline

Run commands from the **repository root** (`AI_OCR/`) with `python src/...`. The pipeline expects split CSVs with aligned rows so **image** and **text** models can be compared fairly; a future **hybrid** model can use the same files and join on `filename` (and `invoice_id` when present).

## 1. Audit the dataset

Read-only: counts classes, missing images, optional `invoice_id` / `semantic_name`, and lists `config` classes with zero examples.

```bash
python src/audit_dataset.py ^
  --labels-csv data/labels_synthetic.csv ^
  --images-dir data/images_synthetic ^
  --output-dir output/audit_synthetic
```

- Add `--force` to overwrite an existing `dataset_audit_summary.json` in that folder.
- Outputs: `dataset_audit_summary.json`, `class_distribution.csv`, `missing_images.csv` (if any), and `document_distribution.csv` if `invoice_id` or `source_pdf` is present.

## 2. Create document-level splits

**Why document-level?** A random **row** split often puts lines from the **same** invoice in both train and validation, so the model is evaluated on nearly the same layout and typography it trained on. That **inflates** accuracy and is a form of **leakage** at the document level. Splitting by `invoice_id` (or `source_pdf`) keeps whole documents in one split.

- Default group: `invoice_id` if present and not all null; else `source_pdf`; or pass `--group-column`.
- Default ratios: 70% / 15% / 15% (train / val / test). Requires at least **three** document groups.
- The original labels CSV is **not** modified.

```bash
python src/create_splits.py ^
  --labels-csv data/labels_synthetic.csv ^
  --output-dir output/splits_70_15_15 ^
  --seed 42
```

Outputs: `train.csv`, `val.csv`, `test.csv`, `split_metadata.json` (including class counts per split).

**Verify no leakage:** Open `split_metadata.json` and confirm disjoint document ids across splits, or `python -c` to assert:

```python
import pandas as pd
from pathlib import Path
b = Path("output/splits_70_15_15")
t, v, e = (pd.read_csv(b / n) for n in ("train.csv", "val.csv", "test.csv"))
col = "invoice_id"  # or source_pdf
assert set(t[col]) & set(v[col]) == set()
assert set(t[col]) & set(e[col]) == set()
assert set(v[col]) & set(e[col]) == set()
print("ok: no document overlap")
```

## 3. Train the image CNN

Uses pre-built `train.csv` / `val.csv` / `test.csv` and the same `images-dir` for all. Saves under `--output-dir`.

```bash
python src/train.py ^
  --train-csv output/splits_70_15_15/train.csv ^
  --val-csv output/splits_70_15_15/val.csv ^
  --test-csv output/splits_70_15_15/test.csv ^
  --images-dir data/images_synthetic ^
  --output-dir output/exp_cnn_run1 ^
  --epochs 10 ^
  --batch-size 32 ^
  --learning-rate 0.001 ^
  --seed 42 ^
  --device auto ^
  --use-class-weights
```

- Omit `--use-class-weights` for uniform class weights in the loss.
- Artifacts: `config.json`, `metrics.json`, `classification_report_*.json`, `confusion_matrix_*.csv`, `model.pth`.

**Legacy** (old behavior: 80% / 20% on `data/labels.csv`, no test set, saves `models/invoice_cnn.pth`):

```bash
python src/train.py --legacy-80-20
```

## 4. Train the text baseline (TF–IDF + linear model)

Same split files; only `text` and `label` are used. **Missing OCR text** is replaced with an empty string (recorded in `config.json`).

```bash
python src/train_text_baseline.py ^
  --train-csv output/splits_70_15_15/train.csv ^
  --val-csv output/splits_70_15_15/val.csv ^
  --test-csv output/splits_70_15_15/test.csv ^
  --output-dir output/exp_text_run1 ^
  --model logistic_regression ^
  --ngram-min 1 ^
  --ngram-max 2 ^
  --seed 42
```

Optional SVM: `--model linear_svc`. Optional TF–IDF cap: `--max-features 20000`. Class reweighting: default `--class-weight balanced`, or `--class-weight none`.

Artifacts: `config.json`, `metrics.json`, reports, confusion matrices, `model.joblib`.

## 5. Compare experiments

**Why macro F1?** For **imbalanced** classes, **accuracy** is dominated by frequent labels (e.g. OTHER). **Macro F1** averages F1 per class, giving rarer classes equal weight in the score and better reflecting how well the model does on the full taxonomy.

```bash
python src/compare_experiments.py ^
  --experiment output/exp_cnn_run1 ^
  --experiment output/exp_text_run1 ^
  --output-csv output/experiment_comparison.csv
```

Or collect every subfolder of a parent that contains `metrics.json`:

```bash
python src/compare_experiments.py --experiments-dir output --output-csv output/experiment_comparison.csv
```

(Only subdirectories with a `metrics.json` are included; put each run in its own folder.)

## 6. Future hybrid extension

- Keep using **one** `create_splits` output so **train/val/test** share identical rows.
- A hybrid `Dataset` can return `(image_tensor, text_str, label)` from the same CSV row (`filename` + `text` + `label`); a fusion module can be added without changing the split utilities or the audit tool.
- `metrics_reporting.py` is shared for comparable evaluation objects.

## Dependencies

Install from the repository root:

```bash
pip install -r requirements.txt
pip install -r synthetic_invoice_generator/requirements.txt
```

(Generator is only needed to **create** PDFs, not to run CNN/text experiments on existing CSVs and images.)
