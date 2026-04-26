# Synthetic invoice generator



Lightweight local utility inside `AI_OCR` that generates synthetic invoice **PDFs** (Jinja2 + WeasyPrint), **per-invoice JSON** ground truth, and a **`manifest.jsonl`** index. Intended for thesis experiments: OCR, line extraction, and future `labels.csv` export—not a product.



**Caption language:** All visible field labels on PDFs are **Polish only** (e.g. *Sprzedawca*, *Nabywca*, *Nr faktury*, *Kwota netto*). Multilingual or mixed Polish/English label modes are **not** used—this keeps synthetic documents consistent and closer to real Polish invoices for OCR and classification experiments. Semantic class ids and JSON structure are unchanged; only the rendered label wording is Polish.



## Layout



- **Code:** `cli.py`, `models.py`, `data_generator.py`, `renderer.py`, `semantic_labels.py`, `label_captions.py`, `hints.py`, `manifest.py`, `paths.py`, `io_utils.py`, `templates_env.py`

- **Templates:** `templates/layout_{a,b,c}.html` + `templates/partials/`

- **Styles:** `static/css/base.css` + per-layout CSS

- **Default outputs:** `out/pdfs/`, `out/json/`, `out/manifest.jsonl` (ignored by git)



## Installation



From the repository root (`AI_OCR`):



```bash

pip install -r synthetic_invoice_generator/requirements.txt

```



Dependencies are limited to Jinja2, WeasyPrint, Faker, and pytest (for tests only if you use a venv for dev).



### Windows (WeasyPrint)



WeasyPrint needs GTK/Pango on Windows. Follow the [WeasyPrint install docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) or use **WSL2** / a Linux environment for batch runs if native install is painful.



## Usage



```bash

cd AI_OCR

python -m synthetic_invoice_generator --count 100 --seed 42 --template any

```



Useful flags:



| Flag | Description |

|------|-------------|

| `--out-dir` | Output root (default: `synthetic_invoice_generator/out`) |

| `--template` | `layout_a`, `layout_b`, `layout_c`, or `any` |

| `--label-locale` | Accepted for compatibility: `pl` (default), or deprecated `en` / `mixed` (ignored; captions and `label_locale` in outputs are always `pl`) |

| `--items-min` / `--items-max` | Line item count bounds |

| `--currency-mode` | `mixed`, `PLN`, `EUR`, or `USD` |

| `--batch-id` | Stable batch name (default derived from seed) |

| `--keep-html` | Save rendered HTML under `out/html/` |

| `--dry-run` | Skip PDF (still writes JSON + manifest; `pdf_sha256` empty) |

| `--fail-fast` | Exit on first error |

| `--overwrite` | Allow reusing an `--out-dir` that already has a manifest and/or PDFs (default: refuse, to avoid accidental append/overwrite) |



## Outputs



### Per-invoice JSON (`out/json/{invoice_id}.json`)



- `schema_version`

- `invoice_id`, `pdf_rel`

- `meta`: `template_id`, `seed`, `batch_id`, `label_locale` (always `pl`), `generator_version`

- `invoice`: full structured invoice (dataclass dump; decimals as strings)

- `render`: captions used for key fields (for traceability; Polish only)

- `classification_hints`: list of `{ semantic, label_id, canonical_value, rendered_value }` aligned with project semantics (see `semantic_labels.py`)

- `line_items_detail`: row-level data for future work (not primary classifier labels)



### `manifest.jsonl`



One JSON object per line: `invoice_id`, `template_id`, `pdf_rel`, `json_rel`, `seed`, `batch_id`, `label_locale` (always `pl`), `page_count`, `pdf_sha256`.



## Relation to OCR / `labels.csv`



This tool does **not** run OCR or emit `filename,text,label` crops. It provides **PDFs + structured truth** so a later step can run docTR (or similar), align lines to `rendered_value` / `canonical_value`, and build `labels.csv` compatible with [`src/config.py`](../src/config.py) plus extended ids in `semantic_labels.py`.



## Semantic / numeric labels



`semantic_labels.py` mirrors `src/config.py` for ids **0–6** and adds **7–11** for `INVOICE_DATE`, `SALE_DATE`, `NET_AMOUNT`, `VAT_AMOUNT`, `CURRENCY`. Keep these in sync when the main project evolves.



## Tests



```bash

cd AI_OCR

pytest synthetic_invoice_generator/tests -q

```



The WeasyPrint PDF test is skipped if WeasyPrint is unavailable or fails to import.

