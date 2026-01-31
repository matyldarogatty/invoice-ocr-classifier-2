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

## Uruchomienie inference
```bash
python src/run_invoice_inference.py data/raw/nowa_faktura.pdf
