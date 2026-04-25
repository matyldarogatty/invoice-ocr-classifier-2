"""Caption pools for invoice field labels (Polish only)."""

from __future__ import annotations

import random
from typing import Dict, List

# Keys used in templates and in ground truth `render` section
CAPTION_KEYS = [
    "invoice_number",
    "invoice_date",
    "sale_date",
    "seller",
    "buyer",
    "seller_nip",
    "buyer_nip",
    "payment_method",
    "payment_due",
    "items_header",
    "net",
    "vat",
    "gross",
    "currency",
    "quantity",
    "unit",
    "unit_net",
    "vat_rate",
    "line_net",
    "line_vat",
    "line_gross",
    "ordinal",
]

# Single vocabulary: Polish variants only (realistic alternates, no English).
_POLISH_POOLS: Dict[str, List[str]] = {
    "invoice_number": ["Nr faktury", "Numer faktury", "Faktura nr"],
    "invoice_date": ["Data wystawienia", "Data wystawienia faktury", "Wystawiono dnia"],
    "sale_date": ["Data sprzedaży", "Data zakończenia dostawy/usług"],
    "seller": ["Sprzedawca", "Wystawca"],
    "buyer": ["Nabywca", "Odbiorca"],
    "seller_nip": ["NIP sprzedawcy", "NIP"],
    "buyer_nip": ["NIP nabywcy", "NIP"],
    "payment_method": ["Metoda płatności", "Forma płatności", "Sposób zapłaty"],
    "payment_due": ["Termin płatności", "Płatne do"],
    "items_header": ["Pozycje faktury", "Wykaz towarów i usług"],
    "net": ["Kwota netto", "Wartość netto", "Netto", "Razem netto"],
    "vat": ["Kwota VAT", "Razem VAT"],
    "gross": ["Kwota brutto", "Wartość brutto", "Razem brutto", "Brutto"],
    "currency": ["Waluta", "Kod waluty"],
    "quantity": ["Ilość", "Il."],
    "unit": ["Jednostka", "J.m."],
    "unit_net": ["Cena netto", "Cena jedn. netto"],
    "vat_rate": ["Stawka VAT", "VAT %"],
    "line_net": ["Wartość netto", "Netto", "Wart. netto"],
    "line_vat": ["Kwota VAT", "VAT wiersza"],
    "line_gross": ["Wartość brutto", "Brutto", "Wart. brutto"],
    "ordinal": ["Lp.", "Lp"],
}


def pick_captions(rng: random.Random) -> Dict[str, str]:
    """Return one Polish caption string per CAPTION_KEYS."""
    return {key: rng.choice(_POLISH_POOLS[key]) for key in CAPTION_KEYS}
