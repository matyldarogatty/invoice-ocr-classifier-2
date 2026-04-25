import json
import random
from pathlib import Path

import pytest

from synthetic_invoice_generator.cli import main
from synthetic_invoice_generator.data_generator import generate_invoice
from synthetic_invoice_generator.hints import build_display_values, build_invoice_document
from synthetic_invoice_generator.label_captions import pick_captions
from synthetic_invoice_generator.manifest import append_manifest


def test_build_invoice_json_schema():
    rng = random.Random(0)
    inv = generate_invoice(
        rng,
        invoice_index=0,
        batch_id="bt",
        seed=0,
        template_id="layout_a",
        label_locale="pl",
        items_min=1,
        items_max=2,
        currency_mode="PLN",
    )
    caps = pick_captions(rng)
    disp = build_display_values(inv, rng)
    doc = build_invoice_document(
        schema_version=1,
        invoice=inv,
        captions=caps,
        display=disp,
        pdf_rel="pdfs/x.pdf",
    )
    assert doc["schema_version"] == 1
    assert doc["meta"]["label_locale"] == "pl"
    assert "classification_hints" in doc
    assert "line_items_detail" in doc
    sem = {h["semantic"] for h in doc["classification_hints"]}
    assert "INVOICE_NUMBER" in sem
    assert "CURRENCY" in sem
    assert "INVOICE_DATE" in sem


def test_manifest_append_line(tmp_path: Path):
    m = tmp_path / "manifest.jsonl"
    append_manifest(m, {"a": 1})
    append_manifest(m, {"b": 2})
    lines = m.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1


def test_cli_dry_run_writes_json_and_manifest(tmp_path: Path):
    code = main(
        [
            "--count",
            "2",
            "--out-dir",
            str(tmp_path),
            "--seed",
            "7",
            "--template",
            "layout_b",
            "--label-locale",
            "pl",
            "--items-min",
            "1",
            "--items-max",
            "2",
            "--currency-mode",
            "PLN",
            "--batch-id",
            "cli_test",
            "--dry-run",
            "--log-level",
            "ERROR",
        ]
    )
    assert code == 0
    manifest = tmp_path / "manifest.jsonl"
    assert manifest.is_file()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["pdf_sha256"] == ""
        jpath = tmp_path / row["json_rel"]
        assert jpath.is_file()
        data = json.loads(jpath.read_text(encoding="utf-8"))
        assert data["meta"]["batch_id"] == "cli_test"
        assert data["meta"]["label_locale"] == "pl"


def test_cli_deprecated_label_locale_writes_pl_metadata(tmp_path: Path):
    code = main(
        [
            "--count",
            "1",
            "--out-dir",
            str(tmp_path),
            "--seed",
            "1",
            "--template",
            "layout_a",
            "--label-locale",
            "mixed",
            "--items-min",
            "1",
            "--items-max",
            "1",
            "--currency-mode",
            "PLN",
            "--dry-run",
            "--log-level",
            "ERROR",
        ]
    )
    assert code == 0
    row = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["label_locale"] == "pl"
    data = json.loads((tmp_path / row["json_rel"]).read_text(encoding="utf-8"))
    assert data["meta"]["label_locale"] == "pl"


def _weasyprint_usable() -> bool:
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@pytest.mark.parametrize("tpl", ["layout_a", "layout_b", "layout_c"])
def test_weasyprint_pdf_smoke(tmp_path: Path, tpl: str):
    if not _weasyprint_usable():
        pytest.skip("WeasyPrint / system libraries not available")
    from synthetic_invoice_generator.data_generator import build_render_helpers, generate_invoice
    from synthetic_invoice_generator.hints import build_display_values
    from synthetic_invoice_generator.label_captions import pick_captions
    from synthetic_invoice_generator.paths import STATIC_DIR
    from synthetic_invoice_generator.renderer import (
        build_html_context,
        html_to_pdf,
        render_invoice_html,
    )

    rng = random.Random(3)

    inv = generate_invoice(
        rng,
        invoice_index=0,
        batch_id="pdf",
        seed=3,
        template_id=tpl,
        label_locale="mixed",
        items_min=1,
        items_max=2,
        currency_mode="PLN",
    )
    assert inv.meta.label_locale == "pl"
    caps = pick_captions(rng)
    disp = build_display_values(inv, rng)
    helpers = build_render_helpers(rng)
    html = render_invoice_html(tpl, build_html_context(inv, caps, helpers, disp))
    pdf_path = tmp_path / "t.pdf"
    html_to_pdf(html, pdf_path, base_url=STATIC_DIR)
    assert pdf_path.stat().st_size > 500
