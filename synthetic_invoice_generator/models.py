"""Invoice dataclasses and consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List


@dataclass
class Address:
    street: str
    postal_code: str
    city: str
    country: str


@dataclass
class Party:
    name: str
    nip: str
    address: Address


@dataclass
class LineItem:
    ordinal: int
    name: str
    quantity: Decimal
    unit: str
    unit_net: Decimal
    vat_rate_percent: int
    line_net: Decimal
    line_vat: Decimal
    line_gross: Decimal


@dataclass
class Totals:
    currency: str
    net: Decimal
    vat: Decimal
    gross: Decimal


@dataclass
class GenerationMeta:
    invoice_id: str
    template_id: str
    seed: int
    batch_id: str
    label_locale: str
    generator_version: str


@dataclass
class InvoiceRecord:
    invoice_number: str
    invoice_date: date
    sale_date: date
    payment_method: str
    payment_due_date: date
    seller: Party
    buyer: Party
    items: List[LineItem]
    totals: Totals
    meta: GenerationMeta


def _decimal_to_json(d: Decimal) -> str:
    return format(d, "f")


def invoice_record_to_dict(inv: InvoiceRecord) -> Dict[str, Any]:
    """JSON-serializable dict tree for InvoiceRecord and nested dataclasses."""

    def conv(x: Any) -> Any:
        if isinstance(x, Decimal):
            return _decimal_to_json(x)
        if isinstance(x, date):
            return x.isoformat()
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        return x

    return conv(asdict(inv))


def assert_invoice_consistent(invoice: InvoiceRecord) -> None:
    """Raise AssertionError if invoice math or dates are inconsistent."""
    if invoice.sale_date > invoice.invoice_date:
        raise AssertionError("sale_date must be on or before invoice_date")

    if not invoice.items:
        raise AssertionError("invoice must have at least one line item")

    sum_net = Decimal("0")
    sum_vat = Decimal("0")
    sum_gross = Decimal("0")
    for it in invoice.items:
        exp_net = (it.quantity * it.unit_net).quantize(Decimal("0.01"))
        if it.line_net != exp_net:
            raise AssertionError(
                f"line {it.ordinal} line_net {it.line_net} != qty*unit_net {exp_net}"
            )
        rate = Decimal(it.vat_rate_percent) / Decimal("100")
        exp_vat = (it.line_net * rate).quantize(Decimal("0.01"))
        if it.line_vat != exp_vat:
            raise AssertionError(
                f"line {it.ordinal} line_vat {it.line_vat} != expected {exp_vat}"
            )
        exp_gross = (it.line_net + it.line_vat).quantize(Decimal("0.01"))
        if it.line_gross != exp_gross:
            raise AssertionError(
                f"line {it.ordinal} line_gross {it.line_gross} != net+vat {exp_gross}"
            )
        sum_net += it.line_net
        sum_vat += it.line_vat
        sum_gross += it.line_gross

    sum_net = sum_net.quantize(Decimal("0.01"))
    sum_vat = sum_vat.quantize(Decimal("0.01"))
    sum_gross = sum_gross.quantize(Decimal("0.01"))

    if invoice.totals.net != sum_net:
        raise AssertionError(f"totals.net {invoice.totals.net} != sum lines {sum_net}")
    if invoice.totals.vat != sum_vat:
        raise AssertionError(f"totals.vat {invoice.totals.vat} != sum lines {sum_vat}")
    if invoice.totals.gross != sum_gross:
        raise AssertionError(
            f"totals.gross {invoice.totals.gross} != sum lines {sum_gross}"
        )

    g = (invoice.totals.net + invoice.totals.vat).quantize(Decimal("0.01"))
    if invoice.totals.gross != g:
        raise AssertionError(f"gross {invoice.totals.gross} != net+vat {g}")

    if not invoice.totals.currency.strip():
        raise AssertionError("currency must be non-empty")

    for party in (invoice.seller, invoice.buyer):
        if not party.name.strip():
            raise AssertionError("party name required")
        if not party.nip.strip():
            raise AssertionError("party nip required")
