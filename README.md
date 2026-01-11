# Invoice OCR & Classification

Projekt demonstracyjny OCR + klasyfikacji elementów faktury
z wykorzystaniem PyTorch.

## Funkcjonalności
- OCR fragmentów faktury
- Klasyfikacja tekstu (NIP, sprzedawca, data, inne)
- Inference na nowych fakturach PDF

## Struktura
- `src/` – kod źródłowy
- `data/` – dane treningowe
- `models/` – wytrenowany model
- `output/` – wyniki predykcji

## Uruchomienie inference
```bash
python src/run_invoice_inference.py data/raw/nowa_faktura.pdf
