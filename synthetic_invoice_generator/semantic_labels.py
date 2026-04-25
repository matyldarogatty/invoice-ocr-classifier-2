"""
Semantic label names and numeric IDs for classification hints in ground truth.

Must match src/config.py LABELS exactly (same semantic string -> same int).
"""

from typing import Dict, Final

# semantic string -> int (aligned with src/config.py LABELS)
SEMANTIC_TO_LABEL_ID: Final[Dict[str, int]] = {
    "SELLER_NAME": 0,
    "SELLER_NIP": 1,
    "BUYER_NAME": 2,
    "BUYER_NIP": 3,
    "TOTAL_PRICE": 4,
    "INVOICE_NUMBER": 5,
    "INVOICE_DATE": 6,
    "SALE_DATE": 7,
    "NET_AMOUNT": 8,
    "VAT_AMOUNT": 9,
    "CURRENCY": 10,
    "OTHER": 11,
}

LABEL_ID_TO_SEMANTIC: Final[Dict[int, str]] = {v: k for k, v in SEMANTIC_TO_LABEL_ID.items()}

GENERATOR_VERSION: Final[str] = "1.0.0"
