"""Synthetic invoice data generation."""

from __future__ import annotations

import random
import string
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Tuple

from faker import Faker

from .models import (
    Address,
    GenerationMeta,
    InvoiceRecord,
    LineItem,
    Party,
    Totals,
    assert_invoice_consistent,
)
from .semantic_labels import GENERATOR_VERSION

VAT_RATES = (0, 5, 8, 23)
UNITS = ("szt.", "kg", "h", "m", "usł.")
PAYMENT_METHODS_PL = ("przelew", "gotówka", "karta", "przelew online", "BLIK")
CURRENCIES_WEIGHTED = (
    [("PLN", 14)] + [("EUR", 2)] + [("USD", 1)]
)  # expanded to flat list below


def _weighted_currency(rng: random.Random) -> str:
    flat: List[str] = []
    for c, w in CURRENCIES_WEIGHTED:
        flat.extend([c] * w)
    return rng.choice(flat)


def _currency_mode_filter(mode: str, rng: random.Random) -> str:
    if mode == "PLN":
        return "PLN"
    if mode == "EUR":
        return "EUR"
    if mode == "USD":
        return "USD"
    return _weighted_currency(rng)


def _synthetic_nip(rng: random.Random, spaced: bool) -> str:
    digits = "".join(rng.choice(string.digits) for _ in range(10))
    if not spaced:
        return digits
    return f"{digits[:3]} {digits[3:6]} {digits[6:8]} {digits[8:]}"


def _invoice_number(rng: random.Random) -> str:
    styles = [
        lambda: f"FV/{rng.randint(2024, 2026)}/{rng.randint(1, 12):02d}/{rng.randint(1, 9999):04d}",
        lambda: f"{rng.randint(100, 999)}/FV/{rng.randint(1, 99999)}",
        lambda: f"FV-{rng.randint(2024, 2026)}-{rng.randint(1, 99999):05d}",
        lambda: f"{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}/{rng.randint(1, 12)}/{rng.randint(2024, 2026)}",
    ]
    return rng.choice(styles)()


def _money(rng: random.Random) -> Decimal:
    whole = rng.randint(1, 5000)
    frac = rng.randint(0, 99)
    return Decimal(f"{whole}.{frac:02d}")


def _format_pl_amount(d: Decimal) -> str:
    s = f"{d:.2f}"
    intp, frac = s.split(".")
    intp = intp[::-1]
    grouped = ".".join(intp[i : i + 3] for i in range(0, len(intp), 3))[::-1]
    return f"{grouped},{frac}"


def _format_date_render(d: date, rng: random.Random) -> str:
    if rng.random() < 0.65:
        return d.strftime("%d.%m.%Y")
    return d.isoformat()


def _nip_canonical(nip: str) -> str:
    return "".join(c for c in nip if c.isdigit())


def generate_line_items(
    rng: random.Random,
    n: int,
    currency: str,
) -> Tuple[List[LineItem], Totals]:
    items: List[LineItem] = []
    for i in range(n):
        qty_int = rng.randint(1, 20)
        qty_frac = rng.choice([Decimal("0"), Decimal("0.5"), Decimal("0.25")])
        quantity = Decimal(qty_int) + qty_frac
        unit = rng.choice(UNITS)
        unit_net = _money(rng)
        rate = rng.choice(VAT_RATES)
        line_net = (quantity * unit_net).quantize(Decimal("0.01"))
        line_vat = (line_net * Decimal(rate) / Decimal("100")).quantize(Decimal("0.01"))
        line_gross = (line_net + line_vat).quantize(Decimal("0.01"))
        name = f"{rng.choice(['Usługa', 'Towar', 'Pozycja', 'Produkt'])} {rng.choice(string.ascii_uppercase)}{rng.randint(10, 99)}"
        items.append(
            LineItem(
                ordinal=i + 1,
                name=name,
                quantity=quantity,
                unit=unit,
                unit_net=unit_net,
                vat_rate_percent=rate,
                line_net=line_net,
                line_vat=line_vat,
                line_gross=line_gross,
            )
        )
    net = sum((x.line_net for x in items), Decimal("0")).quantize(Decimal("0.01"))
    vat = sum((x.line_vat for x in items), Decimal("0")).quantize(Decimal("0.01"))
    gross = sum((x.line_gross for x in items), Decimal("0")).quantize(Decimal("0.01"))
    totals = Totals(currency=currency, net=net, vat=vat, gross=gross)
    return items, totals


def generate_invoice(
    rng: random.Random,
    *,
    invoice_index: int,
    batch_id: str,
    seed: int,
    template_id: str,
    label_locale: str,
    items_min: int,
    items_max: int,
    currency_mode: str,
) -> InvoiceRecord:
    """``label_locale`` is ignored for compatibility; metadata always records ``pl`` (Polish-only captions)."""
    del label_locale
    fake = Faker("pl_PL")
    fake.seed_instance((seed * 1_000_003 + invoice_index) % (2**32))

    currency = _currency_mode_filter(currency_mode, rng)
    n_items = rng.randint(items_min, items_max)
    items, totals = generate_line_items(rng, n_items, currency)

    inv_date = date.today() - timedelta(days=rng.randint(0, 400))
    sale_delta = rng.randint(0, 14)
    sale_date = inv_date - timedelta(days=sale_delta)
    due_days = rng.randint(7, 45)
    payment_due = inv_date + timedelta(days=due_days)

    seller_nip = _synthetic_nip(rng, spaced=rng.random() < 0.4)
    buyer_nip = _synthetic_nip(rng, spaced=rng.random() < 0.5)

    def party(is_seller: bool) -> Party:
        company = fake.company()
        if len(company) < 10 and rng.random() < 0.3:
            company = f"{company} {fake.company_suffix()}"
        nip = seller_nip if is_seller else buyer_nip
        addr = Address(
            street=fake.street_address(),
            postal_code=fake.postcode(),
            city=fake.city(),
            country="Polska",
        )
        return Party(name=company, nip=nip, address=addr)

    invoice_id = f"{batch_id}_{invoice_index:06d}"

    meta = GenerationMeta(
        invoice_id=invoice_id,
        template_id=template_id,
        seed=seed,
        batch_id=batch_id,
        label_locale="pl",
        generator_version=GENERATOR_VERSION,
    )

    inv = InvoiceRecord(
        invoice_number=_invoice_number(rng),
        invoice_date=inv_date,
        sale_date=sale_date,
        payment_method=rng.choice(PAYMENT_METHODS_PL),
        payment_due_date=payment_due,
        seller=party(True),
        buyer=party(False),
        items=items,
        totals=totals,
        meta=meta,
    )
    assert_invoice_consistent(inv)
    return inv


def build_render_helpers(rng: random.Random) -> dict:
    """Jinja helpers: Polish-style amounts and dates (e.g. line items)."""

    def fmt_money(d: Decimal) -> str:
        return _format_pl_amount(Decimal(d))

    def fmt_date(dt: date) -> str:
        return _format_date_render(dt, rng)

    return {"fmt_money": fmt_money, "fmt_date": fmt_date}
