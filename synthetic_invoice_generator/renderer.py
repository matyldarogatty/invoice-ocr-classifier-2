"""HTML render and PDF conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paths import STATIC_DIR
from .templates_env import get_jinja_env, render_template


def build_html_context(invoice, captions: dict, helpers: dict, display: dict) -> dict:
    """Context passed to Jinja templates."""
    return {
        "inv": invoice,
        "c": captions,
        "d": display,
        "fmt_money": helpers["fmt_money"],
        "fmt_date": helpers["fmt_date"],
    }


def render_invoice_html(template_id: str, context: dict) -> str:
    mapping = {
        "layout_a": "layout_a.html",
        "layout_b": "layout_b.html",
        "layout_c": "layout_c.html",
    }
    name = mapping.get(template_id, "layout_a.html")
    env = get_jinja_env()
    return render_template(env, name, context)


def html_to_pdf(html_string: str, pdf_path: Path, base_url: Optional[Path] = None) -> None:
    """Write PDF using WeasyPrint. base_url should point to static/ for CSS."""
    from weasyprint import HTML

    url = str((base_url or STATIC_DIR).resolve().as_uri()) + "/"
    HTML(string=html_string, base_url=url).write_pdf(str(pdf_path))
