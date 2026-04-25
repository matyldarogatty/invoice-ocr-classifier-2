import random

import pytest

from synthetic_invoice_generator.data_generator import generate_invoice
from synthetic_invoice_generator.models import assert_invoice_consistent


def test_invoice_consistency():
    rng = random.Random(123)
    inv = generate_invoice(
        rng,
        invoice_index=0,
        batch_id="t",
        seed=123,
        template_id="layout_a",
        label_locale="pl",
        items_min=2,
        items_max=5,
        currency_mode="PLN",
    )
    assert_invoice_consistent(inv)


def test_sale_before_or_equal_invoice():
    rng = random.Random(0)
    for i in range(30):
        inv = generate_invoice(
            rng,
            invoice_index=i,
            batch_id="t",
            seed=0,
            template_id="layout_b",
            label_locale="mixed",
            items_min=1,
            items_max=8,
            currency_mode="mixed",
        )
        assert inv.sale_date <= inv.invoice_date
        assert inv.meta.label_locale == "pl"


def test_deterministic_core_fields():
    rng1 = random.Random(999)
    rng2 = random.Random(999)
    a = generate_invoice(
        rng1,
        invoice_index=0,
        batch_id="b",
        seed=999,
        template_id="layout_c",
        label_locale="pl",
        items_min=3,
        items_max=3,
        currency_mode="EUR",
    )
    b = generate_invoice(
        rng2,
        invoice_index=0,
        batch_id="b",
        seed=999,
        template_id="layout_c",
        label_locale="pl",
        items_min=3,
        items_max=3,
        currency_mode="EUR",
    )
    assert a.invoice_number == b.invoice_number
    assert a.totals.gross == b.totals.gross
    assert len(a.items) == len(b.items)
