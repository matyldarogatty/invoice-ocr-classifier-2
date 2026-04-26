# Project current state

Reference snapshot for thesis work on **AI_OCR**: invoice line OCR, synthetic data, and **semantic line classification** (image CNN vs TF–IDF text baselines). For **how to run** commands and **preliminary result tables**, see **`README.md`**, **`EXPERIMENTS.md`**, and **`docs/RESULTS_SUMMARY.md`**. For **module-level** detail, see **`docs/TECHNICAL_OVERVIEW.md`**.

---

## 1. Project goal

- **Problem:** Assign a **semantic class** to each **line** of a Polish-style invoice (seller/buyer, NIPs, dates, amounts, OTHER, etc.).
- **Pipeline:** PDF → **docTR** (line boxes + text) → **image crop** + CSV (`filename`, `text`, `label`, …) → **classifier** → label.
- **Compared approaches:** (1) **CNN** on 128×128 grayscale crops; (2) **TF–IDF + Logistic Regression**; (3) **TF–IDF + LinearSVC** on OCR text. **Hybrid** and OCR fine-tuning are **out of scope**.

---

## 2. Architecture (current)

- **Scripts / packages:** `src/` (main logic), `synthetic_invoice_generator/` (PDF+JSON), `data/`, `output/`, optional `docs/`.
- **OCR:** docTR pretrained (not trained in-repo).
- **Synthetic labeling:** `export_synthetic_to_labels.py` + **`match_utils`** (rules, conservative ties → OTHER).
- **Experiments:** **`train.py`**, **`train_text_baseline.py`** with CLI: document-level **split CSVs**, **`--exclude-labels`**, **label remapping** (CNN), **train-only** **`--downsample-label` / `--downsample-ratio`**, seeds, **`metrics_reporting`**, **`compare_experiments.py`**.
- **Taxonomy:** **`src/config.py`** defines **12** classes; **main line-level experiments exclude CURRENCY (id 10)** because export data often has **no** CURRENCY rows (currency appears on amount lines). Exclusion is **in-memory** during training; **ids in config are not removed**.

---

## 3. Data flow (updated)

- **Synthetic:** generator → export → `labels_synthetic.csv` + `images_synthetic/`.
- **Manual:** `OCR.py` → `data/images/` + `data/labels.csv` (overwrites CSV — backup first).
- **Audit:** `audit_dataset.py` (read-only).
- **Splits:** `create_splits.py` / **`splitting.py`** — **document-level** 70/15/15; source CSV **unchanged**.
- **Training:** DataFrames filtered/remapped in memory; **no** overwrite of split CSVs on disk.

Older notes about **row-only** `random_split` in `train.py` apply only to **`--legacy-80-20`**, not to the primary **split-CSV** workflow.

---

## 4. Machine learning pipeline (current)

| Component | Behavior |
|-----------|----------|
| **Active classes** | Typically **11** when **`--exclude-labels 10`**; **`label_mapping.json`** stores original ↔ training indices for CNN. |
| **CNN head** | **`InvoiceCNN(num_classes=…)`** matches active count. |
| **Loss** | Optional **`--use-class-weights`** (balanced on **training** remapped labels). |
| **Metrics** | **Original** label ids for **active** classes; confusion matrices and reports **exclude** dropped classes (no CURRENCY row when excluded). |
| **Text** | Same exclusion/downsampling; sklearn sees **contiguous** training labels internally; reports mapped to **original** ids. |

**Reproducibility:** experiment mode sets seeds (**random**, **numpy**, **torch**) when using the CLI split workflow.

---

## 5. Preliminary findings (single snapshot — not final)

From **`EXPERIMENTS.md`** / **`docs/RESULTS_SUMMARY.md`** ( **100** synthetic documents, **11** active classes):

- **Best validation macro F1** among reported runs: **TF–IDF + LogReg** (~**0.646**); **CNN + class weights** ~**0.632**.
- **Best test accuracy** in that table: **TF–IDF + LinearSVC** (~**0.911**).
- **CNN + train OTHER downsampling** underperformed the CNN **without** downsampling on val/test in that snapshot — not a general conclusion without more runs.

Results must be **repeated** on larger data and multiple seeds before thesis claims.

---

## 6. Inference caveat

**`run_invoice_inference.py`** / **`inference.py`** assume a **full 12-class** **`InvoiceCNN`** unless you change loading code. Checkpoints from **`--exclude-labels`** have **fewer** output neurons and **`label_mapping.json`** — **do not** load them in the stock inference scripts without matching architecture and post-processing.

---

## 7. Implementation status

| Area | Status |
|------|--------|
| Synthetic generator + tests | Yes (`synthetic_invoice_generator/tests/`) |
| Export + audit + document splits | Yes |
| CNN + text training + compare | Yes |
| Label exclusion, remap, train downsample | Yes (`experiment_prep.py`) |
| Root `requirements.txt` | Present (ML stack) |
| Tests for `src/` | No |
| Notebooks | No |
| Hybrid model | No |

---

## 8. Known limitations

- **Synthetic / single-domain**; real scans not covered by default data.
- **OTHER** still dominates; metrics must include **macro F1** (and per-class reports).
- **CURRENCY** excluded from **main** experiment by design — not evidence that currency is “solved.”
- **Environment** may not be fully pinned across machines; document Python and package versions for the thesis.
- **Duplication** of bbox/crop helpers across OCR, export, inference.

---

## 9. Suggested next steps (non-binding)

1. Scale to **300–500+** documents; **multiple** split seeds.
2. Log **commit hash**, exact CLI, and artifact paths for each table in the thesis.
3. Extend **`docs/`** only as needed; avoid duplicating long prose between README, EXPERIMENTS, and RESULTS_SUMMARY.
4. If inference must support **11-class** models, add a **separate** inference path or documented conversion — **not** done in this documentation-only update.

---

*Regenerate this file when the pipeline, data scale, or headline metrics change materially.*
