"""Polish-only caption vocabulary checks."""

import random

import pytest

from synthetic_invoice_generator.label_captions import CAPTION_KEYS, pick_captions

# Former English caption strings (must never appear on generated invoices).
_DEPRECATED_EN_CAPTIONS = frozenset(
    {
        "Invoice Number",
        "Invoice No.",
        "No.",
        "Issue date",
        "Date of issue",
        "Invoice date",
        "Sale date",
        "Supply date",
        "Seller",
        "Supplier",
        "Buyer",
        "Customer",
        "Seller tax ID",
        "Seller NIP",
        "Buyer tax ID",
        "Buyer NIP",
        "Payment method",
        "Payment type",
        "Due date",
        "Payment due",
        "Line items",
        "Goods and services",
        "Net amount",
        "Subtotal (net)",
        "VAT amount",
        "Tax (VAT)",
        "Gross amount",
        "Total (gross)",
        "Currency",
        "CCY",
        "Qty",
        "Quantity",
        "UoM",
        "Unit",
        "Unit price (net)",
        "Net unit price",
        "VAT rate",
        "Tax %",
        "Line net",
        "Line VAT",
        "Line gross",
    }
)


def test_pick_captions_returns_all_keys():
    caps = pick_captions(random.Random(0))
    assert set(caps.keys()) == set(CAPTION_KEYS)


@pytest.mark.parametrize("seed", range(50))
def test_pick_captions_polish_only_no_legacy_english(seed: int):
    caps = pick_captions(random.Random(seed))
    for v in caps.values():
        assert v not in _DEPRECATED_EN_CAPTIONS
        assert v == v.strip()
        assert len(v) > 0
