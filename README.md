# Invoice OCR & Classification

Projekt demonstracyjny OCR + klasyfikacji elementów faktury
z wykorzystaniem PyTorch.

## Funkcjonalności
- OCR fragmentów faktury
- Klasyfikacja tekstu (SELLER_NAME, SELLER_NIP, BUYER_NAME, BUYER_NIP, TOTAL_PRICE, INVOICE_NUMBER, OTHER)
- Inference na nowych fakturach PDF

## Struktura
- `src/` – kod źródłowy
- `data/` – dane treningowe
- `models/` – wytrenowany model
- `output/` – wyniki predykcji
- `synthetic_invoice_generator/` – generator syntetycznych faktur PDF + ground truth (patrz lokalny README)

## Eksport zbioru z faktur syntetycznych (PDF + JSON → crops + CSV)

Po wygenerowaniu plików w `synthetic_invoice_generator/out/` można zbudować **osobny** zbiór (bez nadpisywania `data/labels.csv` ani `data/images/`):

```bash
cd AI_OCR
python src/export_synthetic_to_labels.py --limit 0 --log-level INFO
```

Domyślne wyjścia: `data/images_synthetic/`, `data/labels_synthetic.csv`, `data/labels_synthetic_review.csv`, `data/labels_synthetic_summary.json`.

## Uruchomienie inference
```bash
python src/run_invoice_inference.py data/raw/nowa_faktura.pdf
```
