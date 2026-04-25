import random

from synthetic_invoice_generator.data_generator import build_render_helpers, generate_invoice
from synthetic_invoice_generator.hints import build_display_values
from synthetic_invoice_generator.label_captions import pick_captions
from synthetic_invoice_generator.renderer import build_html_context, render_invoice_html


def test_all_templates_render_html():
    rng = random.Random(1)
    for tid in ("layout_a", "layout_b", "layout_c"):
        inv = generate_invoice(
            rng,
            invoice_index=0,
            batch_id="x",
            seed=1,
            template_id=tid,
            label_locale="mixed",
            items_min=2,
            items_max=2,
            currency_mode="PLN",
        )
        caps = pick_captions(rng)
        disp = build_display_values(inv, rng)
        helpers = build_render_helpers(rng)
        ctx = build_html_context(inv, caps, helpers, disp)
        html = render_invoice_html(tid, ctx)
        assert inv.invoice_number in html
        assert "<table" in html.lower()
