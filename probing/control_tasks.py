"""Hewitt & Liang (2019) control tasks for probe selectivity.

Distinguishes "label is decodable from representations" from "probe is
expressive enough to memorize the label."  A control task assigns each
sample a label that is:

  - deterministic: same input -> same control label across runs
  - structure-preserving: samples sharing the natural equivalence unit
    (e.g. the same lemma, or the same morphosyntactic tag) receive the
    same control label
  - distribution-matched: class marginals match the real labels

selectivity := real_probe_accuracy - control_task_accuracy

A representation that encodes the linguistic property gives a real-vs-control
gap; a probe that simply has enough capacity to memorize a mapping shrinks
that gap.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

# Per-property control-task structure unit.
#   "lemma":     lemma-level properties (every form of a lemma gets one label)
#   "tag":       tag-level properties (every form realising a given target
#                  tag tuple gets one label)
#   "src1_tag":  level of the first source-position tag
#   "src2_tag":  level of the second source-position tag
#   "stem":      level of the stem-final consonant cluster
CONTROL_UNIT = {
    "mood": "tag",
    "tense": "tag",
    "person": "tag",
    "number": "tag",
    "in_l_cell": "tag",
    "l_shaped": "lemma",
    "diphthongization": "lemma",
    "alternation_class": "lemma",
    "conjugation": "lemma",
    "src1_mood": "src1_tag",
    "src1_person": "src1_tag",
    "src2_mood": "src2_tag",
    "src2_person": "src2_tag",
    "stem_final": "stem",
}


def _hash_seed(key: str, salt: str) -> int:
    h = hashlib.blake2b(f"{salt}::{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def make_control_labels(
    units: Sequence[str],
    real_labels: Iterable[int],
    salt: str,
) -> np.ndarray:
    """Generate a deterministic, structure-preserving control label per sample.

    Parameters
    ----------
    units : per-sample equivalence-unit identifier (e.g. lemma string, tag
            string).  Samples sharing a unit are guaranteed the same control
            label.
    real_labels : per-sample real labels; the class marginal is used to draw
            the control labels so that majority-baseline accuracy matches.
    salt : a per-property string mixed into the per-unit seed, so that the
            same lemma can receive different control labels under different
            properties.

    Returns
    -------
    np.ndarray of int64 control labels, same length as units.
    """
    real = np.asarray(list(real_labels))
    classes, counts = np.unique(real, return_counts=True)
    probs = counts / counts.sum()

    unit_to_label: dict[str, int] = {}
    out = np.empty(len(units), dtype=np.int64)
    for i, u in enumerate(units):
        label = unit_to_label.get(u)
        if label is None:
            rng = np.random.RandomState(_hash_seed(u, salt))
            label = int(rng.choice(classes, p=probs))
            unit_to_label[u] = label
        out[i] = label
    return out


def build_units(
    property_name: str,
    *,
    lemma_strs: Sequence[str],
    target_tag_strs: Sequence[str],
    src1_tag_strs: Sequence[str] | None = None,
    src2_tag_strs: Sequence[str] | None = None,
    stem_strs: Sequence[str] | None = None,
) -> Sequence[str]:
    """Return the per-sample equivalence-unit string for the given property."""
    unit_kind = CONTROL_UNIT.get(property_name)
    if unit_kind is None:
        raise KeyError(
            f"No control-task unit defined for property {property_name!r}. "
            f"Add an entry to CONTROL_UNIT before requesting it."
        )
    if unit_kind == "lemma":
        return lemma_strs
    if unit_kind == "tag":
        return target_tag_strs
    if unit_kind == "src1_tag":
        if src1_tag_strs is None:
            raise ValueError("src1_tag_strs is required for src1_* properties")
        return src1_tag_strs
    if unit_kind == "src2_tag":
        if src2_tag_strs is None:
            raise ValueError("src2_tag_strs is required for src2_* properties")
        return src2_tag_strs
    if unit_kind == "stem":
        if stem_strs is None:
            raise ValueError("stem_strs is required for stem_final property")
        return stem_strs
    raise ValueError(f"Unknown control-task unit kind: {unit_kind!r}")
