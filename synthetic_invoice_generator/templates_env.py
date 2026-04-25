"""Jinja2 environment for invoice HTML."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .paths import TEMPLATES_DIR


def get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(
    env: Environment,
    template_name: str,
    context: dict,
) -> str:
    tpl = env.get_template(template_name)
    return tpl.render(**context)
