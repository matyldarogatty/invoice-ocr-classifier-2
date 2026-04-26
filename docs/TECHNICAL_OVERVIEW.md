# Technical overview — modules and data flow

Concise map of the repository for developers. Commands and results are in **`README.md`**, **`EXPERIMENTS.md`**, and **`docs/RESULTS_SUMMARY.md`**.

## End-to-end data flow

```text
synthetic_invoice_generator  →  PDF + JSON + manifest.jsonl
export_synthetic_to_labels   →  docTR OCR + match_utils  →  crops + labels CSV
audit_dataset                →  read-only report
create_splits + splitting    →  train.csv / val.csv / test.csv (by document)
experiment_prep (used inside train*) →  optional exclude labels, remap, train-only downsample
train.py / train_text_baseline.py   →  metrics.json, model checkpoint, label_mapping.json
compare_experiments          →  aggregate metrics.json from several output dirs
```

Manual path: **`OCR.py`** → `data/images/` + `data/labels.csv` (labels often filled offline).

## `src/config.py`

Canonical **`LABELS`** (id → name), **`NUM_CLASSES`**, **`IMG_SIZE`** (128). All training and export logic assumes this mapping; the synthetic generator’s `semantic_labels.py` must stay aligned.

## `src/OCR.py`

Batch script: docTR on PDFs in `data/raw/`, writes line crops to `data/images/`, overwrites **`data/labels.csv`** with `filename`, `text`, empty `label` (manual labeling implied). **Destructive** to existing `labels.csv` if re-run without backup.

## `src/export_synthetic_to_labels.py`

Reads **manifest** + PDFs + JSON; runs docTR; crops lines; calls **`match_ocr_line`**; writes **safe** outputs (refuses default overwrite of `data/labels.csv` / `data/images/`). Optional **`--write-diagnostics`** for currency-token line CSV. Prints class distribution and **`currency_matched_lines`** in summary.

## `src/match_utils.py`

Rule-based alignment of one OCR line string to **`classification_hints`** in JSON (NIPs, amounts, dates, currency tokens, captions → OTHER, ties → OTHER). **No ML**.

## `src/audit_dataset.py`

CLI: load labels CSV + images dir; counts classes, missing images, optional **`invoice_id`** / **`semantic_name`**; writes JSON + CSV summaries; read-only on input data.

## `src/create_splits.py` + `src/splitting.py`

**Document-level** split (default **`invoice_id`**, else **`source_pdf`**): 70/15/15, seed, disjoint documents. Writes **`train.csv`**, **`val.csv`**, **`test.csv`**, **`split_metadata.json`**. Does not edit the source labels file.

## `src/experiment_prep.py`

Shared helpers: **exclude** label ids from DataFrames, **remap** original ids → contiguous training ids for the CNN, **train-only** downsampling of one label (e.g. OTHER) by ratio × max count of other classes. Used by **`train.py`** and **`train_text_baseline.py`** only.

## `src/invoice_dataset.py`

PyTorch **`Dataset`**: loads image by **`filename`**, grayscale resize 128×128, returns tensor + **label**. Accepts **`csv_path`** or in-memory **`dataframe`** (for filtered/remapped experiments).

## `src/model.py`

**`InvoiceCNN`**: conv blocks + MLP. **`num_classes`** defaults to **`NUM_CLASSES`**; set to **active class count** when labels are excluded so **`CrossEntropyLoss`** matches logits.

## `src/train.py`

CLI training: split CSVs, **`--exclude-labels`**, **`--downsample-label`** + **`--downsample-ratio`** (train only), **`--use-class-weights`**, seeds, device, **`label_mapping.json`**, metrics in **original id** space for active classes. **`--legacy-80-20`** for old single-file behavior.

## `src/train_text_baseline.py`

TF–IDF + **LogisticRegression** or **LinearSVC**; same exclusion / downsampling / mapping story as CNN; saves **`model.joblib`**.

## `src/metrics_reporting.py`

**`compute_split_metrics`**, **`confusion_matrix_to_csv`** (supports **`label_order`** for non-contiguous original ids), **`save_json`**.

## `src/compare_experiments.py`

Reads **`metrics.json`** from each **`--experiment`** dir (or subdirs of **`--experiments-dir`**) and writes a comparison CSV.

## `src/run_invoice_inference.py` + `src/inference.py`

**`run_invoice_inference.py`:** docTR on one PDF → crops → **`InvoiceCNN`** (default **12** outputs) → prints/saves by predicted class.  
**`inference.py`:** single crop image → CNN.

**Caution:** checkpoints trained with **`--exclude-labels`** have **K ≠ 12** logits. Loading them in scripts that assume full **`NUM_CLASSES`** will **not** match without code changes and mapping logic.

## `synthetic_invoice_generator/`

**CLI** (`python -m synthetic_invoice_generator`): **`--count`**, **`--seed`**, **`--out-dir`**, templates, **`--overwrite`** to reuse a non-empty output dir. Jinja2 + WeasyPrint + Faker → PDFs, JSON ground truth, **`classification_hints`**. See package **`README.md`**.

---

*File list is not exhaustive (`OCR_infer.py`, `export` helpers, tests under `synthetic_invoice_generator/tests/` exist but are secondary for the main thesis pipeline.)*
