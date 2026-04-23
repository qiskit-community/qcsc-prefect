from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape


def make_env(pkg: str) -> Environment:
    """
    pkg: e.g. "qcsc_prefect_adapters.miyabi"
    """
    env = Environment(
        loader=PackageLoader(pkg, "templates"),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
        undefined=StrictUndefined,
    )
    return env
