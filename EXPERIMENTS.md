# Experiment pipeline

All commands assume the **repository root** (`AI_OCR/`) as the current working directory. Use `python src/...` (PowerShell-friendly; line continuation with `` ` `` is optional).

## Overview: end-to-end flow

1. **Synthetic invoice generation** — `python -m synthetic_invoice_generator` → PDFs, JSON, `manifest.jsonl`.
2. **Synthetic label export** — `export_synthetic_to_labels.py` → line crops + `labels_synthetic.csv` (docTR + `match_utils`).
3. **Dataset audit** — `audit_dataset.py` → counts, missing files, class coverage.
4. **Document-level split** — `create_splits.py` → `train.csv` / `val.csv` / `test.csv` (no overlap of `invoice_id` / `source_pdf`).
5. **CNN and/or text baselines** — `train.py`, `train_text_baseline.py` (optional `--exclude-labels 10`, optional train-only OTHER downsampling).
6. **Compare runs** — `compare_experiments.py` → CSV from multiple `metrics.json`.

The same split files let you compare **image** vs **text** fairly (aligned rows). A **hybrid** model is not implemented.

---

## 1. Audit the dataset

Read-only on input CSV and image directory.

```powershell
python src/audit_dataset.py --labels-csv data/labels_synthetic.csv --images-dir data/images_synthetic --output-dir output/audit_synthetic --force
```

- Add `--force` to overwrite existing `dataset_audit_summary.json` in that output folder.
- Outputs: `dataset_audit_summary.json`, `class_distribution.csv`, optional `missing_images.csv`, optional `document_distribution.csv`.

---

## 2. Create document-level splits

**Why document-level?** A random **row** split puts lines from the **same** invoice in train and val, which inflates metrics (layout/font leakage). Splitting by **`invoice_id`** (or **`source_pdf`**) keeps whole documents in one split.

- Ratios default **70% / 15% / 15%**; requires at least **three** documents.
- Source CSV is **not** modified.

```powershell
python src/create_splits.py --labels-csv data/labels_synthetic.csv --output-dir output/splits_70_15_15 --seed 42
```

Outputs: `train.csv`, `val.csv`, `test.csv`, `split_metadata.json`.

**Leakage check** (disjoint document ids):

```python
import pandas as pd
from pathlib import Path
b = Path("output/splits_70_15_15")
t, v, e = (pd.read_csv(b / n) for n in ("train.csv", "val.csv", "test.csv"))
col = "invoice_id"
assert set(t[col]) & set(v[col]) == set()
assert set(t[col]) & set(e[col]) == set()
assert set(v[col]) & set(e[col]) == set()
print("ok: no document overlap")
```

---

## 3. Train the image CNN

Uses split CSVs + shared `images-dir`. Artifacts under `--output-dir`: `config.json`, `metrics.json`, `label_mapping.json` (when excluding labels), optional `sampling_summary.json`, classification reports, confusion matrices, `model.pth`.

**Reference run — class weights, CURRENCY excluded (id 10):**

```powershell
python src/train.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_no_currency_weighted_run1 --epochs 10 --batch-size 32 --device cpu --seed 42 --exclude-labels 10 --use-class-weights
```

**Same + train-only OTHER downsampling** (`11` = OTHER, cap = ratio × max count of non-OTHER in **train**):

```powershell
python src/train.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_no_currency_weighted_downsampled_run1 --epochs 10 --batch-size 32 --device cpu --seed 42 --exclude-labels 10 --use-class-weights --downsample-label 11 --downsample-ratio 3.0
```

**Legacy** (80/20 on `data/labels.csv`, no test set):

```powershell
python src/train.py --legacy-80-20
```

---

## 4. Train text baselines (TF–IDF + linear models)

Missing OCR `text` is replaced with an empty string (recorded in `config.json`). Default sklearn **`class_weight`** is balanced unless `--class-weight none`.

**Logistic Regression:**

```powershell
python src/train_text_baseline.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --output-dir output/exp_text_logreg_no_currency_run1 --exclude-labels 10
```

**Linear SVC:**

```powershell
python src/train_text_baseline.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --output-dir output/exp_text_svm_no_currency_run1 --exclude-labels 10 --model linear_svc
```

Optional: `--max-features`, `--downsample-label 11 --downsample-ratio 3.0` (train only).

---

## 5. Compare experiments

```powershell
python src/compare_experiments.py --experiment output/exp_cnn_no_currency_weighted_run1 --experiment output/exp_cnn_no_currency_weighted_downsampled_run1 --experiment output/exp_text_logreg_no_currency_run1 --experiment output/exp_text_svm_no_currency_run1 --output-csv output/experiment_comparison_final_run1.csv
```

Or scan subfolders that contain `metrics.json`:

```powershell
python src/compare_experiments.py --experiments-dir output --output-csv output/experiment_comparison.csv
```

---

## 6. Class imbalance, OTHER, macro F1, excluding CURRENCY

- **OTHER** dominates typical synthetic exports (e.g. ~**83%** of lines in a 100-document snapshot: **3876 / 4656**). **Accuracy** can stay high if the model predicts OTHER often; it is a weak sole metric.
- **Macro F1** averages F1 over **active** classes with equal weight — better for comparing CNN vs text under imbalance. Report both macro and weighted F1 from `metrics.json`.
- **CURRENCY (id 10)** had **0** labeled rows in that snapshot: currency usually sits on **amount** lines (`Razem: … PLN`), not as a stable separate class in the export. The **main** experiments use **`--exclude-labels 10`**: rows removed **in memory only**; **`config.LABELS`** and ids are **unchanged**.
- **Train-only OTHER downsampling** reduces dominant-class gradients on **train**; **val/test** stay natural. Whether it helps is an **empirical** question (see preliminary table below).

---

## 7. Synthetic generation and export

**Generate** (new `--out-dir` per batch, or `--overwrite` only if you accept reuse):

```powershell
python -m synthetic_invoice_generator --count 100 --seed 42 --out-dir synthetic_output/batch_100
```

**Export** (adjust paths to your manifest):

```powershell
python src/export_synthetic_to_labels.py --manifest synthetic_output/batch_100/manifest.jsonl --pdf-dir synthetic_output/batch_100/pdfs --json-dir synthetic_output/batch_100/json --images-dir data/images_synthetic --csv-path data/labels_synthetic.csv --review-csv-path data/labels_synthetic_review.csv
```

Optional: `--write-diagnostics` for OCR lines containing currency tokens. Default export **blocks** writing into `data/labels.csv` and `data/images/`.

---

## 8. Excluding labels and downsampling (summary)

- **`--exclude-labels 10`** — drops CURRENCY rows from train/val/test **in memory**; CNN gets **`num_classes = 11`** with **remapping** (`label_mapping.json`); metrics use **original** ids for active classes only.
- **`--downsample-label 11 --downsample-ratio 3.0`** — **train** only: cap OTHER at `3.0 ×` (max class count among non-OTHER in train). Requires **`--seed`**. Both flags required together.

See **`docs/TECHNICAL_OVERVIEW.md`** for module-level detail.

---

## 9. Current preliminary results (100 synthetic documents)

**These are not final thesis results.** They come from **one** protocol: **100 documents**, **4656** rows, **0** missing labels/text/images, **OTHER 3876**, **CURRENCY 0** before exclusion, **11 active classes** in runs below (`--exclude-labels 10`). Splits and training use **seed 42** where documented. **Repeat** on larger data and **multiple seeds** before drawing strong conclusions. **Do not commit** large generated datasets or `output/` run folders to git unless required.

| Model | Input | Variant | val_accuracy | val_macro_f1 | val_weighted_f1 | test_accuracy | test_macro_f1 | test_weighted_f1 |
|--------|--------|---------|--------------|--------------|-----------------|---------------|---------------|------------------|
| InvoiceCNN | image | class weights, no CURRENCY | 0.9072 | **0.6324** | 0.9076 | 0.8987 | **0.4197** | 0.9005 |
| InvoiceCNN | image | class weights + train-only OTHER downsample (×3), no CURRENCY | 0.7938 | 0.5632 | 0.8345 | 0.6076 | 0.4159 | 0.7148 |
| Tfidf_LogReg | OCR text | no CURRENCY | 0.9485 | **0.6463** | 0.9343 | 0.8861 | 0.3887 | 0.8896 |
| Tfidf_LinearSVC | OCR text | no CURRENCY | **0.9485** | 0.6034 | 0.9269 | **0.9114** | 0.3901 | **0.9030** |

**Reading the table**

- **Best val macro F1** in this snapshot: **TF–IDF + LogReg** (0.646); **CNN + class weights** is close (0.632).
- **Best test accuracy**: **LinearSVC** (0.911). Accuracy favors frequent classes; combine with **macro F1** and per-class reports.
- **CNN + train OTHER downsampling**: lower val/test accuracy here; investigate before treating downsampling as default.

Short narrative copy: **`docs/RESULTS_SUMMARY.md`**.

---

## 10. Dependencies

```powershell
pip install -r requirements.txt
pip install -r synthetic_invoice_generator/requirements.txt
```

---

## 11. Future hybrid extension

Same split CSVs could feed a dataset that returns `(image, text, label)` for a later fusion model; not implemented.
