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
- Artifacts: `config.json`, `metrics.json`, `label_mapping.json` (when using `--exclude-labels`), optional `sampling_summary.json` (train downsampling), `classification_report_*.json`, `confusion_matrix_*.csv`, `model.pth`.

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

Artifacts: `config.json`, `metrics.json`, `label_mapping.json` (if excluding labels), optional `sampling_summary.json`, reports, confusion matrices, `model.joblib`.

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

---

## 7. Preparing a larger synthetic dataset

**Why CURRENCY (and rarer fields) need examples in the export:** The classifier and TF–IDF model both learn from **row-level** labels. If a class has **zero** rows, there is no supervised signal for that class; the export matcher must link OCR lines to the JSON `CURRENCY` hint (often: a **standalone** `zł` or `PLN` line, not only text inside a total line, which can tie with `TOTAL_PRICE` and become **OTHER**).

**Why 10 documents is only a smoke test:** A **document-level** split with few invoices yields tiny val/test, unstable class counts, and macro F1 that does **not** generalize. Use **hundreds** of generated invoices for serious runs; treat small runs as pipeline checks only.

**Why OTHER dominates and accuracy misleads:** The **OTHER** class is frequent; high **accuracy** can mean “predict OTHER always.” **Macro F1** (in `metrics.json`) is the headline metric for comparing image vs. text on the same splits.

**Generator safety (no silent overwrite):** By default, `python -m synthetic_invoice_generator` **refuses** to write into an `--out-dir` that **already** has a non-empty `manifest.jsonl` and/or files under `pdfs/`. Use a **new** directory for each batch, or pass **`--overwrite`** if you accept appending to the manifest and reusing the same folder.

### 7.1 Generate many synthetic PDFs + JSON (example: 100 or 500)

```bash
cd AI_OCR
python -m synthetic_invoice_generator --count 100 --seed 42 --out-dir synthetic_output/batch_100
```

- Repeat with a **different** `--out-dir` for another batch, or the same path **with** `--overwrite` only if you understand manifest append/replace.
- For final thesis-scale runs, use e.g. `--count 500` and a new output folder name (and enough disk / time).

### 7.2 Export PDFs to line crops and `labels_synthetic.csv`

```bash
python src/export_synthetic_to_labels.py ^
  --manifest synthetic_output/batch_100/manifest.jsonl ^
  --pdf-dir synthetic_output/batch_100/pdfs ^
  --json-dir synthetic_output/batch_100/json ^
  --images-dir data/images_synthetic ^
  --csv-path data/labels_synthetic.csv ^
  --review-csv-path data/labels_synthetic_review.csv
```

- Use a **separate** `--images-dir` / `--csv-path` (or a new folder) if you do not want to replace an existing export; the script blocks writing over `data/labels.csv` and `data/images/` by default.
- **Optional** `--write-diagnostics` writes `currency_candidate_lines.csv` (or `--diagnostics-csv`), listing OCR lines that contain PLN / zł / EUR / USD tokens for review.
- After export, check logs for **“CURRENCY — matched …”** and **“Final class distribution:”**.

### 7.3 Audit → splits → CNN → text → compare (same as sections 1–5)

Re-run the commands from this file, pointing at your **new** `labels_synthetic.csv` and **images** directory, e.g.:

```bash
python src/audit_dataset.py --labels-csv data/labels_synthetic.csv --images-dir data/images_synthetic --output-dir output/audit_synthetic --force
python src/create_splits.py --labels-csv data/labels_synthetic.csv --output-dir output/splits_70_15_15 --seed 42
python src/train.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_run1 --epochs 10 --device auto --seed 42
python src/train_text_baseline.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --output-dir output/exp_text_run1
python src/compare_experiments.py --experiment output/exp_cnn_run1 --experiment output/exp_text_run1 --output-csv output/experiment_comparison.csv
```

**Smoke v2 (quick checks, not thesis results):** use distinct output folder names, e.g. `output/audit_synthetic_v2`, `output/splits_70_15_15_v2`, `output/exp_cnn_smoke_v2` (1 epoch), `output/exp_text_smoke_v2`, and `output/experiment_comparison_smoke_v2.csv`.

---

## 8. Excluding CURRENCY and optional train-only OTHER downsampling

**Why exclude CURRENCY (id `10`) from the main experiment:** On real exports, currency often appears **on the same OCR line** as an amount (e.g. `Razem: 1230,00 PLN`), not as a separate line class. The export matcher frequently maps those lines to **TOTAL_PRICE** / **OTHER**, so **CURRENCY** can have **no or almost no** examples. Training a 12-way classifier with an empty class is misleading. **`config.LABELS` and numeric ids are unchanged**; exclusion is **only** applied in memory during training/eval for that run.

**Why not remove OTHER from val/test:** Evaluation should reflect the **natural** line distribution in held-out documents. Dropping OTHER from validation or test would make scores **unrealistic** for deployment.

**Why downsample OTHER only on train:** Reducing **OTHER** in **training** balances gradients and can help the model learn rarer fields; val/test stay **unmodified** so reported metrics still reflect real imbalance.

**Macro F1 vs accuracy:** With heavy OTHER, **accuracy** can stay high while rare classes are ignored. **Macro F1** (in `metrics.json`) averages over **active** classes only and is better for comparing CNN vs text on the same protocol.

**Behavior:**

- `--exclude-labels 10` — drop rows whose `label` is `10` from train, val, and test **in memory** (CSV files on disk are **not** modified). Multiple ids: `--exclude-labels 10 11` (space-separated). Omit the flag for all classes in `config`.
- The CNN remaps remaining original ids to **contiguous** `0 … K-1` for `CrossEntropyLoss`; **`label_mapping.json`** records `original_to_training` / `training_to_original`. Metrics and confusion matrices use **original** ids and names for **active** classes only.
- `--downsample-label 11 --downsample-ratio 3.0` — **train split only**: keep at most `3.0 × (max count among all non-11 labels)` rows with label `11`. Requires the experiment **seed**. Both flags must be passed together. Written to `config.json` and, when applied, **`sampling_summary.json`**.

### Example commands (replace paths with your split folder)

CNN smoke without CURRENCY:

```bash
python src/train.py --train-csv output/splits_70_15_15_v2/train.csv --val-csv output/splits_70_15_15_v2/val.csv --test-csv output/splits_70_15_15_v2/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_smoke_no_currency --epochs 1 --device auto --seed 42 --exclude-labels 10
```

CNN smoke without CURRENCY + train-only OTHER downsampling:

```bash
python src/train.py --train-csv output/splits_70_15_15_v2/train.csv --val-csv output/splits_70_15_15_v2/val.csv --test-csv output/splits_70_15_15_v2/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_smoke_no_currency_downsampled --epochs 1 --device auto --seed 42 --exclude-labels 10 --downsample-label 11 --downsample-ratio 3.0
```

Text baseline without CURRENCY:

```bash
python src/train_text_baseline.py --train-csv output/splits_70_15_15_v2/train.csv --val-csv output/splits_70_15_15_v2/val.csv --test-csv output/splits_70_15_15_v2/test.csv --output-dir output/exp_text_smoke_no_currency --exclude-labels 10
```

Text baseline without CURRENCY + train-only OTHER downsampling:

```bash
python src/train_text_baseline.py --train-csv output/splits_70_15_15_v2/train.csv --val-csv output/splits_70_15_15_v2/val.csv --test-csv output/splits_70_15_15_v2/test.csv --output-dir output/exp_text_smoke_no_currency_downsampled --exclude-labels 10 --downsample-label 11 --downsample-ratio 3.0
```

**Extra outputs:** `label_mapping.json`; optionally `sampling_summary.json` if downsampling ran; `metrics.json` includes `active_original_labels` and `metrics_label_space: original_ids_active_only`.
