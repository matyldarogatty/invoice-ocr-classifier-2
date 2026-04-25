"""Paths relative to the synthetic_invoice_generator package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PACKAGE_DIR / "out"
STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATES_DIR = PACKAGE_DIR / "templates"

PDF_SUBDIR = "pdfs"
JSON_SUBDIR = "json"
HTML_SUBDIR = "html"
MANIFEST_FILENAME = "manifest.jsonl"
