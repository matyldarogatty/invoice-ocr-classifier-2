# Invoice OCR — line-level semantic classification

Master’s thesis project: **semantic classification of invoice line regions** (Polish-style invoices). The system extracts **lines** from PDFs with OCR, builds **image crops** and **OCR text** per line, then assigns a **semantic label** (seller, NIPs, dates, amounts, OTHER, etc.).

## Research goal

Compare **image-based** vs **text-based** line classifiers on the same document-level train/validation/test splits, under heavy **class imbalance** (especially OTHER). The taxonomy remains defined in `src/config.py` (12 classes); the **main** line-level experiments **exclude CURRENCY** (id `10`) because it rarely appears as a separate labeled line in synthetic exports.

## High-level pipeline

```text
PDF invoice → docTR OCR → line bounding boxes + OCR text
           → line image crops (+ CSV: filename, text, label, …)
           → classifier → semantic label per line
```

## Approaches implemented

| Modality | Model | Script |
|----------|--------|--------|
| **Image** | Small CNN on 128×128 grayscale crops | `src/train.py` |
| **OCR text** | TF–IDF + **Logistic Regression** | `src/train_text_baseline.py` (`--model logistic_regression`) |
| **OCR text** | TF–IDF + **Linear SVC** | `src/train_text_baseline.py` (`--model linear_svc`) |

OCR for extraction is **docTR** (pretrained), not fine-tuned here. A **hybrid** model is not implemented.

## Folder structure (short)

| Path | Role |
|------|------|
| `src/` | Config, datasets, training, export, audit, splits, metrics, inference |
| `data/` | `raw/`, `images/`, `images_synthetic/`, `labels*.csv` (defaults vary; many paths gitignored) |
| `synthetic_invoice_generator/` | PDF + JSON generation (`python -m synthetic_invoice_generator`) |
| `models/` | Legacy / ad hoc checkpoints (often gitignored) |
| `output/` | Audits, splits, experiment runs (`metrics.json`, `model.pth`, …) — typically gitignored |
| `docs/` | Optional technical overview and results summary |
| `EXPERIMENTS.md` | Full experiment protocol and **preliminary** result tables |

## Quick start

From the **repository root** (paths below assume `AI_OCR/` as cwd).

**Install**

```powershell
pip install -r requirements.txt
pip install -r synthetic_invoice_generator/requirements.txt
```

**Synthetic data → export → audit → splits** (see `EXPERIMENTS.md` for details)

```powershell
python -m synthetic_invoice_generator --count 100 --seed 42 --out-dir synthetic_output/batch_100
python src/export_synthetic_to_labels.py --manifest synthetic_output/batch_100/manifest.jsonl --pdf-dir synthetic_output/batch_100/pdfs --json-dir synthetic_output/batch_100/json --images-dir data/images_synthetic --csv-path data/labels_synthetic.csv --review-csv-path data/labels_synthetic_review.csv
python src/audit_dataset.py --labels-csv data/labels_synthetic.csv --images-dir data/images_synthetic --output-dir output/audit_synthetic --force
python src/create_splits.py --labels-csv data/labels_synthetic.csv --output-dir output/splits_70_15_15 --seed 42
```

**Train CNN (example: exclude CURRENCY, class weights)**

```powershell
python src/train.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_no_currency_weighted_run1 --epochs 10 --batch-size 32 --device cpu --seed 42 --exclude-labels 10 --use-class-weights
```

**Inference on a PDF** (expects a full 12-class checkpoint compatible with `run_invoice_inference.py`; models trained with `--exclude-labels` have **fewer** outputs — see `EXPERIMENTS.md` / `PROJECT_CURRENT_STATE.md`)

```powershell
python src/run_invoice_inference.py data/raw/nowa_faktura.pdf
```

Full command list, exclusion/downsampling, and **preliminary results**: **`EXPERIMENTS.md`**.

## Where data and outputs live

- **Generated synthetic PDFs/JSON:** e.g. `synthetic_output/…` or `synthetic_invoice_generator/out/` (often gitignored).
- **Exported crops + `labels_synthetic.csv`:** commonly `data/images_synthetic/`, `data/labels_synthetic.csv` (may be gitignored or local-only).
- **Experiment artifacts:** `output/<run>/` — `config.json`, `metrics.json`, `label_mapping.json`, `model.pth` or `model.joblib`, confusion matrices, reports.

## Do not commit

Do **not** commit large generated assets or run outputs unless your course explicitly requires it:

- Synthetic PDFs, bulk PNG crops, full `labels_synthetic.csv` at scale  
- `output/` experiment folders, personal `models/*.pth` from partial-class runs  

Use `.gitignore` (already excludes typical paths) and keep a short note in the thesis of **commit hash + commands** used to reproduce numbers.

---

*Polish notes in older revisions were replaced by this English README for onboarding; the codebase and `config` semantics are unchanged.*
