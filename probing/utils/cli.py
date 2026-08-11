"""docopt helpers shared by the analysis scripts.

Each script's module docstring carries its own ``Usage:`` block; this module
turns the parsed option dict into an attribute namespace (``--model-type`` ->
``args.model_type``), applies type casts, and validates closed choice sets,
which docopt itself does not enforce.
"""

from types import SimpleNamespace

from docopt import docopt


def parse(doc, types=None, choices=None, sentinels=None, version=None):
    """Parse the calling script's docstring with docopt.

    Args:
        doc: the module docstring (must contain a Usage: block).
        types: {attr_name: callable} casts applied when the value is not None.
        choices: {attr_name: iterable} closed sets validated after casting.
        sentinels: {placeholder: real_value} substitutions applied to string
            options. Used for defaults that docopt cannot express, e.g. the
            FEATURE_INFORMED_DATA placeholder standing in for a path computed
            at runtime.
        version: optional version string for --version.

    Returns:
        SimpleNamespace with kebab-case options mapped to snake_case
        attributes and <positionals> mapped to bare names.
    """
    raw = docopt(doc, version=version)
    ns = SimpleNamespace()
    for key, value in raw.items():
        attr = key.lstrip("-<").rstrip(">").replace("-", "_")
        if isinstance(value, str) and value in (sentinels or {}):
            value = sentinels[value]
        setattr(ns, attr, value)
    for attr, cast in (types or {}).items():
        value = getattr(ns, attr, None)
        if value is not None and not isinstance(value, bool):
            setattr(ns, attr, cast(value))
    for attr, allowed in (choices or {}).items():
        value = getattr(ns, attr, None)
        if value is not None and value not in allowed:
            raise SystemExit(
                f"invalid --{attr.replace('_', '-')}: {value!r} (choose from {sorted(allowed)})")
    return ns


def standard_sentinels():
    """The path placeholders shared by the analysis scripts."""
    import os
    from probing import FEATURE_INFORMED_ROOT
    return {
        "FEATURE_INFORMED_DATA": os.path.join(FEATURE_INFORMED_ROOT, "data"),
        "FEATURE_INFORMED_SEPCHAR": os.path.join(FEATURE_INFORMED_ROOT, "data", "seperate_char_data"),
        "FEATURE_INFORMED_DATABIN": os.path.join(FEATURE_INFORMED_ROOT, "data", "char_sep_databin_aligned"),
    }
