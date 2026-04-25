# Project Current State

This document describes the **current** state of the AI_OCR repository as a reference for academic work (e.g., a master’s thesis). It is derived from inspection of the source tree, configuration, data layout, and scripts. It does not prescribe future design choices.

---

## 1. Project goal

The project addresses **semantic classification of invoice line regions** in Polish business documents. The intended workflow is:

1. **Segmentation at line level** using a pretrained optical character recognition (OCR) stack (docTR) on PDF pages.
2. **Classification of each line crop** into a fixed set of semantic categories (e.g., seller name, NIP, invoice number, dates, amounts, currency, or a catch-all “other” class).

The machine-learning contribution in this repository is a **supervised image classifier** (a small convolutional neural network, CNN) that maps **grayscale 128×128 image patches** to class labels. Text produced by OCR is used for **dataset construction and logging**, not as an input feature to the CNN in the current implementation.

A complementary component generates **synthetic invoice PDFs** and structured **JSON ground truth** for controlled experiments, with an export path that labels line crops by **rule-based matching** between OCR output and known field values in JSON.

**Inputs:** PDF invoice files; for training, CSV files listing image filenames and integer labels, plus corresponding image files.

**Outputs:** Trained model weights (`invoice_cnn.pth`); at inference, predicted class per line, optional saving of crops under per-class directories; for data preparation, crops and label CSVs.

---

## 2. Architecture

The codebase is **script-oriented**: there is no HTTP API, no packaged application entry point, and no Jupyter notebooks in the repository. Two subsystems interact:

| Subsystem | Technology | Role |
|-----------|------------|------|
| **OCR and line extraction** | docTR (`ocr_predictor`, pretrained) | Renders the page, outputs blocks/lines, yields line bounding boxes and recognized text. |
| **Line classification** | PyTorch `InvoiceCNN` | Consumes a single-channel tensor per crop; outputs logits over `NUM_CLASSES` (12). |
| **Synthetic data** | Jinja2, WeasyPrint, Faker (`synthetic_invoice_generator` package) | Produces PDFs, per-invoice JSON, and a manifest. |
| **Synthetic labeling** | `export_synthetic_to_labels.py` + `match_utils.py` | Runs OCR on each synthetic PDF, assigns labels by conservative string/value matching to `classification_hints` in JSON. |

**Directory roles (high level):**

- `src/` — Application logic: config, dataset, model, training, inference, synthetic export, OCR batch scripts.
- `data/` — Training and derived assets: `raw/` (PDFs, gitignored in default config), `images/` and `images_synthetic/`, `labels.csv`, `labels_synthetic*.csv`, review and summary files.
- `synthetic_invoice_generator/` — Python package with CLI (`python -m synthetic_invoice_generator`), tests under `synthetic_invoice_generator/tests/`, templates and static assets.
- `models/` — Intended location for `invoice_cnn.pth` (weight files are gitignored).
- `output/` — Training metrics (`output/debug/`) and inference outputs (`output/predictions/`), typically gitignored.

**Semantic class definitions** are centralized in `src/config.py` (`LABELS`, `NUM_CLASSES`, `IMG_SIZE`). The generator’s `synthetic_invoice_generator/semantic_labels.py` is required to stay aligned with those numeric identifiers.

**Main entry points (operational):**

- Batch OCR and crop export for manually labeled data: `src/OCR.py`.
- Model training: `src/train.py`.
- PDF inference (OCR + CNN): `src/run_invoice_inference.py`.
- Single-image classification: `src/inference.py`.
- Synthetic export: `src/export_synthetic_to_labels.py`.
- Invoice generation: `python -m synthetic_invoice_generator` (from repository root, with appropriate `PYTHONPATH` or install).

---

## 3. Data flow

### 3.1 Real (manual) pipeline

1. Place PDFs under `data/raw/`.
2. Run `OCR.py`: docTR processes each PDF, crops each line to `data/images/`, and writes `data/labels.csv` with columns `filename`, `text`, and `label` (the script initially leaves `label` empty; labels are assumed to be filled manually or by an external process).
3. `train.py` loads `data/labels.csv` and `data/images/` via `InvoiceDataset` (Pandas, PIL, torchvision transforms: grayscale, resize, tensor).

**Risk:** `OCR.py` overwrites `data/labels.csv` on each run, which can destroy prior manual labels if not backed up.

### 3.2 Synthetic pipeline

1. Run the synthetic generator to produce `out/pdfs/`, `out/json/`, and `out/manifest.jsonl` (default under `synthetic_invoice_generator/out/`, often gitignored).
2. Run `export_synthetic_to_labels.py` with default or explicit paths. The script:
   - Refuses to write to `data/labels.csv` or `data/images/`.
   - For each manifest entry, loads PDF and JSON, runs docTR, crops lines, and calls `match_ocr_line` in `match_utils.py` to map OCR text to a semantic class using `classification_hints` and caption heuristics.
3. Writes crops (e.g., to `data/images_synthetic/`) and `labels_synthetic.csv` (with extra columns such as `invoice_id`, `semantic_name`), plus review CSV and optional JSON summary.

### 3.3 Preprocessing and features

- **Preprocessing:** Bounding boxes smaller than a fixed pixel threshold are skipped; images are converted to grayscale, resized to 128×128, and normalized to tensors. No data augmentation is implemented in the shown dataset class.
- **Features:** The classifier uses **raw pixels** of the line crop, not explicit text embeddings or hand-crafted features.

### 3.4 Data leakage and evaluation caveats

- `train.py` uses `random_split` on **individual rows** without grouping by document or `invoice_id`. **Lines from the same invoice can appear in both training and validation**, which can inflate validation accuracy because layout and font statistics are shared across lines.
- There is **no separate test** split in the training script.
- If training labels for synthetic data are produced with the **same** OCR model as at inference, **errors are correlated**; independent evaluation (e.g., different OCR, human review, or held-out real scans) is not automated in the repository.

---

## 4. Machine learning pipeline

### 4.1 Models

- **OCR:** docTR `ocr_predictor(pretrained=True)` — not fine-tuned in this repository.
- **Classifier:** `InvoiceCNN` in `src/model.py` — a shallow CNN (two conv blocks with pooling) and a two-layer MLP head, output size `NUM_CLASSES` (12).

### 4.2 Training

- **Dataset (as wired in `train.py`):** `data/labels.csv` and `data/images/` (paths resolved relative to the project root).
- **Split:** 80% train / 20% validation via `random_split` (unstratified).
- **Optimizer:** Adam, learning rate 0.001.
- **Loss:** `CrossEntropyLoss`.
- **Epochs:** 10 (hardcoded).
- **Batch size:** 32.
- **Device:** Training script sets `device = "cpu"` (GPU use is commented out). Inference in `run_invoice_inference.py` can use CUDA if available.

### 4.3 Validation and metrics

- Validation **accuracy** per epoch.
- **Confusion matrix** and **classification report** (scikit-learn) over validation predictions.
- Per-epoch JSON files under `output/debug/metrics_epoch_*.json`.
- Final weights saved to `models/invoice_cnn.pth` (path relative to the process working directory when the script is run).

### 4.4 Reproducibility and configuration

- **Randomness:** The training script does not set a global random seed for PyTorch, NumPy, or the train/validation split, so **runs are not bit-reproducible** unless the environment imposes external seeding.
- **Hyperparameters** (epochs, learning rate, split ratio) are **hardcoded** in `train.py` rather than read from a config file or CLI.

### 4.5 Empirical class coverage (as of documentation time)

The following observations are **data-dependent** and should be re-checked after any re-labeling or re-export.

- The file `data/labels.csv` (on the order of hundreds of lines) is **heavily skewed** toward a single label value (e.g., majority in class index 6 in a measured distribution). Only a **subset of the 12 classes** appears in that file; **several class indices from `config.LABELS` have zero examples** in the measured CSV, so the softmax over 12 classes is **not fully supervised** for all categories on that dataset.
- The synthetic `labels_synthetic.csv` sample examined shows a **dominant** “OTHER” class and uneven coverage; some semantic classes (e.g., **CURRENCY**) may have **very few or zero** rows in a given export, depending on matching success and volume.

These effects impact the interpretability of accuracy and the reliability of per-class performance.

---

## 5. Current implementation status

| Area | Status |
|------|--------|
| Class taxonomy and image size | Implemented in `src/config.py`. |
| PDF → line crops (manual pipeline) | Implemented in `src/OCR.py`. |
| PyTorch `Dataset` and CNN | Implemented (`invoice_dataset.py`, `model.py`). |
| Training loop with validation metrics | Implemented (`train.py`). |
| End-to-end PDF inference | Implemented (`run_invoice_inference.py`). |
| Single image inference | Implemented (`inference.py`). |
| Synthetic invoice generation | Implemented as a package with CLI and **pytest** tests. |
| Synthetic → labeled crops + CSV | Implemented (`export_synthetic_to_labels.py`, `match_utils.py`) with safety checks against overwriting core `data/` files. |
| Committed trained weights | **Not** present in the repository (`.gitignore` excludes `models/*.pth`); a fresh clone requires training or external weights. |
| Root-level ML dependency file | **Absent**; only `synthetic_invoice_generator/requirements.txt` is present for a subset of dependencies. |
| Notebooks | **None** in the repository. |
| Unit/integration tests for `src/` | **None**; tests exist only under `synthetic_invoice_generator/tests/`. |

**Documentation:** The root `README.md` describes structure, synthetic export, and inference. It does **not** document training commands, dependency versions, or experiment protocol in a single place.

---

## 6. Known problems

1. **Environment specification:** No single pinned requirement set at the repository root for PyTorch, torchvision, docTR, pandas, scikit-learn, and NumPy; reproducing a historical environment is difficult.
2. **Data overwrite risk:** `OCR.py` overwrites `data/labels.csv` and regenerates all crops, endangering manual labels.
3. **Code duplication:** Line bounding-box and cropping logic is repeated across multiple modules (OCR, inference, export, ad hoc `OCR_infer.py`).
4. **Path sensitivity:** Many paths are relative; `train.py` mixes `Path` for data with string paths for `output/` and `models/` that depend on the **current working directory** when the script is launched.
5. **Training vs synthetic data:** `train.py` is wired to `data/labels.csv` and `data/images/`, not by default to the synthetic paths; using synthetic data for training requires explicit path changes or file placement.
6. **Class balance and label semantics:** Measured class distributions are highly imbalanced; for the manual CSV, the dominant class index may not align with readers’ expectations from the **names** in `LABELS` (e.g., if many non-date lines share the same index). This must be resolved for sound evaluation.
7. **Validation methodology:** Row-level random split and lack of a test set limit claims about **generalization to new invoices** or new layouts.
8. **Reproducibility:** No training seeds; non-deterministic split between runs.
9. **Ad hoc script:** `OCR_infer.py` uses a **hardcoded** PDF file name; it appears to be a local experiment, not a maintained interface.

---

## 7. Missing elements

- Central **Python version** and **dependency lockfile** (or `requirements.txt` / `pyproject.toml`) for the full ML stack.
- **Unit tests** for training, dataset loading, export, and inference in `src/`.
- **Stratified** or **document-grouped** splitting and a **held-out test** set.
- **Class-weighting**, oversampling, or other strategies for imbalanced and missing classes, if the thesis targets fair per-class metrics.
- **Configuration management** (CLI or YAML) for training and evaluation hyperparameters and paths.
- **Unified module** for shared OCR/geometry code to avoid drift between scripts.
- **Thesis-oriented experiment log** (single JSON or table summarizing data version, commit hash, seeds, metrics) — not present as an automated artifact.
- **Notebooks** or a short methods appendix would need to be added if the program requires them for presentation.

---

## 8. Next recommended tasks

These are **documentation-level recommendations** only; they do not change the codebase by themselves.

1. **Lock the environment** — Add a root-level dependency specification and record the **Python** version used for all reported results.
2. **Stabilize data workflows** — Back up or version `labels.csv`; avoid destructive overwrites in documented procedures.
3. **Clarify the experimental protocol** — Decide whether the primary evaluation is **synthetic-only**, **real-only**, or **mixed**, and document split strategy (e.g., by `invoice_id`).
4. **Align labels with the thesis** — Reconcile `config.LABELS` names with actual annotation rules; address **missing classes** in training data or reduce the class set in the model.
5. **Reproducibility** — Set and document seeds; persist split indices or `invoice_id` lists for each fold.
6. **Metrics** — Report **per-class** precision/recall or macro-F1 alongside accuracy, given imbalance.
7. **Path and interface cleanup** (when the codebase is allowed to change) — Single CLI for train/eval, resolve `train.py`’s `models/` / `output/` relative to the project root, and document the exact `cwd` for every command.
8. **Thesis text** — Add diagrams: end-to-end pipeline, synthetic data generation + hint matching, and a clear **limitations** subsection on OCR cascade and document-level leakage.

---

*This file describes the state of the project at the time of writing. Regenerate or amend it after major changes to data, scripts, or dependencies.*
