"""docopt helpers shared by the analysis scripts.

Turns each script's parsed option dict into an attribute namespace, applies
type casts, and validates closed choice sets, which docopt does not enforce.
"""

from types import SimpleNamespace

from docopt import docopt


def parse(doc, types=None, choices=None, sentinels=None, version=None):
    """Parse doc with docopt into a SimpleNamespace (snake_case attrs; sentinels fill runtime defaults)."""
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
