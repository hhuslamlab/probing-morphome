"""Extract morphological property labels from test data.

Usage:
  extract_labels.py --split SPLIT --run RUN [--data-dir DIR] [--output-dir DIR]
                    [--control-task] [--control-salt SALT]
  extract_labels.py (-h | --help)

Options:
  --split SPLIT        Data split, e.g. 10L_90NL.
  --run RUN            Run identifier, e.g. 1_1.
  --data-dir DIR       Base data directory [default: FEATURE_INFORMED_DATA].
  --output-dir DIR     Output directory for label files [default: data/probing/labels].
  --control-task       Also emit *_control_labels.pt: structure-preserving random
                       labels per Hewitt & Liang (2019) for probe selectivity.
  --control-salt SALT  Salt mixed into the per-unit RNG seed for control-task
                       labels [default: phase1-control-v1].
"""

import json
import logging
import os
import re
import sys
from collections import Counter

import numpy as np
import torch

from probing import SPLITS, PROPERTIES, EXIT_SUCCESS, EXIT_ERROR, EXIT_SKIPPED
from probing.control_tasks import (
    CONTROL_UNIT,
    build_units,
    make_control_labels,
)
from probing.utils.cli import parse, standard_sentinels

logger = logging.getLogger("probing.labels")

# Label encodings
MOOD_MAP = {"IND": 0, "SBJV": 1}
TENSE_MAP = {"PRS": 0}
PERSON_MAP = {"1": 0, "2": 1, "3": 2}
NUMBER_MAP = {"PL": 0, "SG": 1}
LSHAPED_MAP = {"L": 0, "NL": 1}
# Diphthongization is the second Spanish morphome (stress-conditioned e->ie,
# o->ue).  Encoded binary at the lemma level, mirroring l_shaped, so it can
# serve as the within-paper replication target in the falsifiability design.
DIPHTHONG_MAP = {"DIPH": 0, "NODIPH": 1}
ALTERNATION_MAP = {"none": 0, "s_sk": 1, "n_ng": 2, "c_x": 3, "other_l": 4}
IN_L_CELL_MAP = {"in": 0, "out": 1}
CONJUGATION_MAP = {"ar": 0, "er": 1, "ir": 2}
# Per-sample (not lemma-level): do all the surface forms involved in one
# inflection instance -- the source form(s) plus the target form -- share the
# same stem-final consonant cluster?  'differ' marks an alternation across the
# three forms, the surface footprint of the L-shape; 'same' marks none.
STEMFINAL_MATCH_MAP = {"same": 0, "differ": 1}

LABEL_ENCODINGS = {
    "mood": MOOD_MAP,
    "tense": TENSE_MAP,
    "person": PERSON_MAP,
    "number": NUMBER_MAP,
    "l_shaped": LSHAPED_MAP,
    "diphthongization": DIPHTHONG_MAP,
    "alternation_class": ALTERNATION_MAP,
    "in_l_cell": IN_L_CELL_MAP,
    "conjugation": CONJUGATION_MAP,
    "stem_final_match": STEMFINAL_MATCH_MAP,
    "src1_mood": MOOD_MAP,
    "src2_mood": MOOD_MAP,
    "src1_person": PERSON_MAP,
    "src2_person": PERSON_MAP,
}

# Inflectional suffixes for Spanish present tense (-ar, -er, -ir conjugations)
_SUFFIXES = sorted(set([
    "o", "as", "a", "amos", "ajs", "an",       # -ar indicative
    "e", "es", "emos", "ejs", "en",             # -ar subjunctive / -er indicative
    "imos", "is",                                # -ir indicative
]), key=len, reverse=True)

# Paradigm cells inside the L-shape (IND;1;SG + all SBJV)
_L_CELLS = frozenset({
    "V;IND;PRS;1;SG",
    "V;SBJV;PRS;1;SG", "V;SBJV;PRS;2;SG", "V;SBJV;PRS;3;SG",
    "V;SBJV;PRS;1;PL", "V;SBJV;PRS;2;PL", "V;SBJV;PRS;3;PL",
})


def parse_args():
    return parse(__doc__,
                 choices=dict(split=SPLITS),
                 sentinels=standard_sentinels())


def build_lemma_lookup(data_dir):
    """Build (tag, normalized_form) -> lemma string lookup.

    The lemma is the dictionary key under which a given form is stored in the
    paradigm dicts; we use it as the equivalence unit for lemma-level control
    tasks (l_shaped, alternation_class, conjugation).
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    lookup = {}
    for lemma, forms in {**l_dict, **nl_dict}.items():
        for tag, form in forms.items():
            lookup.setdefault((tag, form.replace(" ", "")), lemma)
    return lookup


def build_lshaped_lookup(data_dir):
    """Build a (tag, normalized_form) -> is_lshaped lookup from both paradigm dicts.

    Labels at the lemma level: a sample is L-shaped if its lemma belongs to an
    L-shaped paradigm, regardless of which specific cell is being inflected.
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    lookup = {}  # (tag, normalized_form) -> bool (True = L-shaped)
    for lemma_forms in l_dict.values():
        for tag, form in lemma_forms.items():
            lookup[(tag, form.replace(" ", ""))] = True
    for lemma_forms in nl_dict.values():
        for tag, form in lemma_forms.items():
            lookup.setdefault((tag, form.replace(" ", "")), False)

    return lookup, len(l_dict), len(nl_dict)


def _get_stem(form):
    """Strip inflectional suffix to get the stem."""
    for s in _SUFFIXES:
        if form.endswith(s):
            return form[: -len(s)]
    return None


def _get_stem_final(stem):
    """Extract the final consonant cluster from a stem (ignoring stress marks)."""
    if stem is None:
        return None
    clean = stem.replace("\u02c8", "").replace("\u02cc", "")  # remove ˈ ˌ
    m = re.search(r"([^aeiou]*)$", clean)
    return m.group(1) if m else ""


def _segment_form(segment):
    """Extract the normalized (space-free) IPA form from one src segment.

    Source segments interleave the IPA characters with the morphological tag,
    e.g. "s u p e ɾ b e n ɡ ˈ a m o s V SBJV PRS 1 PL".  IPA glyphs/diacritics
    are lowercase; tag tokens are uppercase or digits, so the form is every
    token up to the first tag token.
    """
    form_toks = []
    for tok in segment.split():
        if tok.isupper() or tok.isdigit():
            break
        form_toks.append(tok)
    return "".join(form_toks)


def _form_stem_final(form):
    """Stem-final consonant cluster of a normalized (space-free) form.

    Mirrors the fallback in _diphthongizes: if no known inflectional suffix
    strips, use the whole form, so the result is always a (possibly empty)
    string rather than None.
    """
    return _get_stem_final(_get_stem(form) or form)


def stemfinal_match_label(src_line, tgt_line):
    """Per-sample binary label: do the forms in one inflection instance share
    the same stem-final consonant cluster?

    The instance's forms are the source form(s) in the .src line plus the
    target form in the .tgt line -- the "three forms" in the dual-source setup
    (src1, src2, target).  Returns STEMFINAL_MATCH_MAP['same'] (0) if all share
    one stem-final cluster, else ['differ'] (1).  Generalizes to any number of
    source forms; the last " # "-delimited src field is the target tag, not a
    form, and is dropped.
    """
    *src_segments, _target_tag = src_line.split(" # ")
    stem_finals = [_form_stem_final(_segment_form(seg)) for seg in src_segments]
    stem_finals.append(_form_stem_final(tgt_line.replace(" ", "")))
    same = all(sf == stem_finals[0] for sf in stem_finals)
    return STEMFINAL_MATCH_MAP["same"] if same else STEMFINAL_MATCH_MAP["differ"]


# Diphthongization detection ------------------------------------------------
# Spanish diphthongization is stress-conditioned: the mid stem vowels e/o
# surface as the rising diphthongs ie [je] / ue [we] in rhizotonic (stem-
# stressed) present cells, but stay monophthongal in arrhizotonic (ending-
# stressed) cells.  We detect it by contrasting a stem-stressed cell with an
# ending-stressed cell of the same lemma.
#
# In the normalised IPA, the diphthong is a glide (j/w) adjacent to the stress
# mark ˈ and a mid vowel (e/o): e.g. "pjˈenso", "kwˈento".  A non-alternating
# lexical glide (present in *both* stressed and unstressed stems) is not
# diphthongization, so we require the glide+vowel to appear in the rhizotonic
# stem and be *absent* from the arrhizotonic stem.
_DIPH_STRESSED_RE = re.compile(r"[jw]ˈ[eo]|ˈ[jw][eo]")
_GLIDE_MID_RE = re.compile(r"[jw][eo]")

# Rhizotonic (stem-stressed) and arrhizotonic (ending-stressed) present cells,
# in preference order; the first present in a paradigm is used.
_RHIZO_TAGS = (
    "V;IND;PRS;3;SG", "V;IND;PRS;1;SG", "V;IND;PRS;2;SG", "V;IND;PRS;3;PL",
)
_ARRHIZO_TAGS = ("V;IND;PRS;1;PL", "V;IND;PRS;2;PL")


def _diphthongizes(rhizo_form, arrhizo_form):
    """True if the lemma shows stress-conditioned e->ie / o->ue diphthongization."""
    if not rhizo_form or not arrhizo_form:
        return False
    r_norm = rhizo_form.replace(" ", "")
    a_norm = arrhizo_form.replace(" ", "")
    r_stem = _get_stem(r_norm) or r_norm
    a_stem = _get_stem(a_norm) or a_norm
    return bool(_DIPH_STRESSED_RE.search(r_stem)) and not bool(_GLIDE_MID_RE.search(a_stem))


def build_diphthong_lookup(data_dir):
    """Build a (tag, normalized_form) -> is_diphthongizing lookup (lemma-level).

    Returns (lookup, n_diphthong_lemmas, example_lemmas).  The label is a
    binary morphome flag mirroring l_shaped; example_lemmas is a small sample
    of detected diphthongizers for sanity-checking the (heuristic) detector.
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    lookup = {}
    n_diph = 0
    examples = []
    for lemma, forms in {**l_dict, **nl_dict}.items():
        rhizo = next((forms[t] for t in _RHIZO_TAGS if t in forms), None)
        arrhizo = next((forms[t] for t in _ARRHIZO_TAGS if t in forms), None)
        diph = _diphthongizes(rhizo, arrhizo)
        if diph:
            n_diph += 1
            if len(examples) < 15:
                examples.append(lemma)
        for tag, form in forms.items():
            lookup.setdefault((tag, form.replace(" ", "")), diph)
    return lookup, n_diph, examples


def _classify_alternation(out_sfs, in_sfs):
    """Classify the consonant alternation type from stem-final consonants.

    Args:
        out_sfs: stem-final consonants from out-of-L cells
        in_sfs: stem-final consonants from in-L cells

    Returns:
        One of: 'none', 's_sk', 'n_ng', 'c_x', 'other_l'
    """
    if not out_sfs or not in_sfs:
        return "none"
    out_sf = Counter(out_sfs).most_common(1)[0][0]
    in_sf = Counter(in_sfs).most_common(1)[0][0]
    if out_sf == in_sf:
        return "none"
    if out_sf == "s" and in_sf == "sk":
        return "s_sk"
    if out_sf == "n" and in_sf == "n\u0261":  # nɡ
        return "n_ng"
    # ç→x backing (with any preceding consonant context)
    if out_sf.endswith("\u00e7") and in_sf.endswith("x") and out_sf[:-1] == in_sf[:-1]:
        return "c_x"
    return "other_l"


def build_alternation_lookup(data_dir):
    """Build a (tag, normalized_form) -> alternation_class lookup.

    For each L-shaped lemma, determines the consonant alternation type by
    comparing stem-final consonants in in-L vs out-of-L paradigm cells.
    Non-L-shaped lemmas are labeled 'none'.

    Classes:
        none    - no consonant alternation (NL lemmas, or L with vowel-only alternation)
        s_sk    - s→sk velar insertion (141 lemmas)
        n_ng    - n→nɡ velar insertion (53 lemmas)
        c_x     - ç→x backing (49 lemmas)
        other_l - other rare consonant alternations (36 lemmas)
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    # Determine alternation class for each L-shaped lemma
    lemma_alt = {}
    for lemma, forms in l_dict.items():
        out_sfs, in_sfs = [], []
        for tag, form in forms.items():
            sf = _get_stem_final(_get_stem(form.replace(" ", "")))
            if sf is not None:
                (in_sfs if tag in _L_CELLS else out_sfs).append(sf)
        lemma_alt[lemma] = _classify_alternation(out_sfs, in_sfs)

    # Build (tag, form) -> alternation_class lookup
    lookup = {}
    for lemma, forms in l_dict.items():
        cls = lemma_alt[lemma]
        for tag, form in forms.items():
            lookup[(tag, form.replace(" ", ""))] = cls
    for lemma_forms in nl_dict.values():
        for tag, form in lemma_forms.items():
            lookup.setdefault((tag, form.replace(" ", "")), "none")

    class_counts = Counter(lemma_alt.values())
    return lookup, class_counts


def _get_conjugation_class(lemma):
    """Determine conjugation class (-ar/-er/-ir) from lemma string.

    Lemma is a space-separated IPA string ending in 'ɾ'. The vowel before
    the final 'ɾ' determines the class.
    """
    clean = lemma.replace(" ", "").replace("\u02c8", "")  # strip spaces + stress
    if clean.endswith("a\u027e") or clean.endswith("a\u0072"):  # aɾ or ar
        return "ar"
    if clean.endswith("e\u027e") or clean.endswith("e\u0072"):  # eɾ or er
        return "er"
    if clean.endswith("i\u027e") or clean.endswith("i\u0072"):  # iɾ or ir
        return "ir"
    return None


def build_stem_final_lookup(data_dir):
    """Build (tag, normalized_form) -> stem_final consonant cluster lookup.

    Used both as a probe target and — more importantly — as a *phonological
    control* for the morphome question: after residualising the
    representations against stem_final, can we still decode L-shape /
    alternation class / conjugation?  If yes, those features are encoded
    above and beyond their surface phonological correlates.
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    raw_lookup: dict[tuple[str, str], str] = {}
    for lemma_forms in {**l_dict, **nl_dict}.values():
        for tag, form in lemma_forms.items():
            normalized = form.replace(" ", "")
            sf = _get_stem_final(_get_stem(normalized))
            raw_lookup.setdefault((tag, normalized), sf if sf is not None else "")

    classes = sorted({v for v in raw_lookup.values()})
    class_map = {sf: i for i, sf in enumerate(classes)}
    lookup = {k: class_map[v] for k, v in raw_lookup.items()}
    return lookup, class_map


def build_conjugation_lookup(data_dir):
    """Build a (tag, normalized_form) -> conjugation_class lookup.

    Determines the conjugation class (-ar, -er, -ir) from each lemma's
    infinitive ending, then maps all forms of that lemma to the class.
    """
    l_path = os.path.join(data_dir, "ipa_clean_lshaped_dict.json")
    nl_path = os.path.join(data_dir, "ipa_clean_non_lshaped_dict.json")
    with open(l_path) as f:
        l_dict = json.load(f)
    with open(nl_path) as f:
        nl_dict = json.load(f)

    lookup = {}
    class_counts = Counter()
    for lemma, forms in {**l_dict, **nl_dict}.items():
        conj = _get_conjugation_class(lemma)
        if conj is None:
            continue
        class_counts[conj] += 1
        for tag, form in forms.items():
            lookup.setdefault((tag, form.replace(" ", "")), conj)

    return lookup, class_counts


def parse_tag(tag_str):
    """Parse a tag like '<V;SBJV;PRS;2;PL>' into (mood, tense, person, number)."""
    inner = tag_str.strip("<>")
    fields = inner.split(";")
    # fields: [V, mood, tense, person, number]
    return {
        "mood": fields[1],
        "tense": fields[2],
        "person": fields[3],
        "number": fields[4],
    }


_TAG_RE = re.compile(r"<([^>]+)>$")


def extract_source_tag(form_tag_str):
    """Extract the tag from a 'form <tag>' string like 's u p e ɾ ... <V;SBJV;PRS;1;PL>'."""
    m = _TAG_RE.search(form_tag_str)
    if m:
        return "<" + m.group(1) + ">"
    return None


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    output_path = os.path.join(args.output_dir, f"{args.split}_{args.run}_labels.pt")
    if os.path.exists(output_path):
        logger.info("Skipping %s_%s -- labels already extracted", args.split, args.run)
        sys.exit(EXIT_SKIPPED)

    run_num = args.run.split("_")[0]
    model_name = f"{args.split}_{args.run}"
    src_path = os.path.join(
        args.data_dir, args.split, "test", f"run{run_num}", f"test.{model_name}.src"
    )
    tgt_path = os.path.join(
        args.data_dir, args.split, "test", f"run{run_num}", f"test.{model_name}.tgt"
    )

    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            logger.error("Missing file: %s", path)
            sys.exit(EXIT_ERROR)

    lshaped_lookup, n_l_lemmas, n_nl_lemmas = build_lshaped_lookup(args.data_dir)
    logger.info(
        "Built L-shaped lookup: %d L lemmas, %d NL lemmas, %d (tag, form) entries",
        n_l_lemmas, n_nl_lemmas, len(lshaped_lookup),
    )

    diphthong_lookup, n_diph_lemmas, diph_examples = build_diphthong_lookup(args.data_dir)
    logger.info(
        "Built diphthongization lookup: %d diphthongizing lemmas detected; examples: %s",
        n_diph_lemmas, diph_examples,
    )

    alternation_lookup, alt_class_counts = build_alternation_lookup(args.data_dir)
    logger.info("Alternation classes (L-shaped lemmas): %s", dict(alt_class_counts))

    conjugation_lookup, conj_class_counts = build_conjugation_lookup(args.data_dir)
    logger.info("Conjugation classes: %s", dict(conj_class_counts))

    # Used for control-task equivalence units.
    lemma_lookup = build_lemma_lookup(args.data_dir)
    logger.info("Built lemma lookup: %d (tag, form) entries", len(lemma_lookup))

    # Phonological control for the morphome probe.
    stem_final_lookup, stem_final_class_map = build_stem_final_lookup(args.data_dir)
    logger.info(
        "Built stem_final lookup: %d classes, %d (tag, form) entries",
        len(stem_final_class_map), len(stem_final_lookup),
    )

    with open(src_path) as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path) as f:
        tgt_lines = [line.strip() for line in f]

    if len(src_lines) != len(tgt_lines):
        logger.error(
            "Line count mismatch: %d src vs %d tgt", len(src_lines), len(tgt_lines)
        )
        sys.exit(EXIT_ERROR)

    n_samples = len(src_lines)
    logger.info("Processing %d samples from %s", n_samples, model_name)

    labels = {prop: [] for prop in PROPERTIES}

    # Per-sample equivalence-unit strings, used only for control-task generation
    units = {
        "lemma": [],
        "target_tag": [],
        "src1_tag": [],
        "src2_tag": [],
        "stem": [],
    }

    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        # Parse source line: [form1 <tag1>] # [form2 <tag2>] # <target_tag>
        parts = src_line.split(" # ")
        target_tag = parts[-1].strip()
        tag_fields = parse_tag(target_tag)

        # Extract source pair tags for primacy/recency probes
        src1_tag_str = extract_source_tag(parts[0])
        src2_tag_str = extract_source_tag(parts[1])
        src1_fields = parse_tag(src1_tag_str) if src1_tag_str else None
        src2_fields = parse_tag(src2_tag_str) if src2_tag_str else None

        # Encode target morphological properties
        for prop in ("mood", "tense", "person", "number"):
            value = tag_fields[prop]
            encoding_map = LABEL_ENCODINGS[prop]
            if value not in encoding_map:
                logger.error(
                    "Unknown %s value '%s' at line %d", prop, value, i + 1
                )
                sys.exit(EXIT_ERROR)
            labels[prop].append(encoding_map[value])

        # Source-position probes (primacy/recency bias)
        for src_idx, src_fields in [("src1", src1_fields), ("src2", src2_fields)]:
            if src_fields is None:
                logger.error("Could not parse %s tag at line %d", src_idx, i + 1)
                sys.exit(EXIT_ERROR)
            labels[f"{src_idx}_mood"].append(MOOD_MAP[src_fields["mood"]])
            labels[f"{src_idx}_person"].append(PERSON_MAP[src_fields["person"]])

        # L-shaped and alternation class (both lemma-level)
        target_tag_bare = target_tag.strip("<>")
        normalized_form = tgt_line.replace(" ", "")
        key = (target_tag_bare, normalized_form)
        if key in lshaped_lookup:
            is_l = lshaped_lookup[key]
            labels["l_shaped"].append(LSHAPED_MAP["L"] if is_l else LSHAPED_MAP["NL"])
        else:
            logger.warning("No lemma match for (tag=%s, form=%s) at line %d", target_tag_bare, normalized_form, i + 1)
            labels["l_shaped"].append(LSHAPED_MAP["NL"])

        # Diphthongization (lemma-level, binary) — the second morphome
        is_diph = diphthong_lookup.get(key, False)
        labels["diphthongization"].append(
            DIPHTHONG_MAP["DIPH"] if is_diph else DIPHTHONG_MAP["NODIPH"]
        )

        alt_cls = alternation_lookup.get(key, "none")
        labels["alternation_class"].append(ALTERNATION_MAP[alt_cls])

        # In-L cell position: is the target cell inside the L-shape?
        labels["in_l_cell"].append(
            IN_L_CELL_MAP["in"] if target_tag_bare in _L_CELLS else IN_L_CELL_MAP["out"]
        )

        # Conjugation class (-ar/-er/-ir)
        conj = conjugation_lookup.get(key, "ar")  # fallback to -ar (dominant class)
        labels["conjugation"].append(CONJUGATION_MAP[conj])

        # Stem-final consonant cluster — the phonological control for the
        # morphome probe.
        sf_label = stem_final_lookup.get(key)
        if sf_label is None:
            # Fallback: compute on the fly from the target form
            sf = _get_stem_final(_get_stem(normalized_form))
            sf_label = stem_final_class_map.get(sf if sf is not None else "", 0)
        labels["stem_final"].append(sf_label)

        # Equivalence-unit strings for control tasks
        units["lemma"].append(lemma_lookup.get(key, f"_unknown_lemma_{i}"))
        units["target_tag"].append(target_tag_bare)
        units["src1_tag"].append(src1_tag_str or f"_unknown_src1_{i}")
        units["src2_tag"].append(src2_tag_str or f"_unknown_src2_{i}")
        stem = _get_stem(normalized_form)
        units["stem"].append(_get_stem_final(stem) or f"_unknown_stem_{i}")

    label_tensors = {prop: torch.tensor(vals, dtype=torch.long) for prop, vals in labels.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        **label_tensors,
        "label_encodings": LABEL_ENCODINGS,
        "stem_final_class_map": stem_final_class_map,
        "n_samples": n_samples,
        # Per-sample grouping keys for leakage-free probe CV.  Grouping folds by
        # lemma stops a probe from "decoding" a lemma-level property (l_shaped,
        # diphthongization, conjugation, ...) by memorising which lemma a form
        # belongs to.
        "groups": {
            "lemma": units["lemma"],
            "stem": units["stem"],
            "target_tag": units["target_tag"],
        },
    }
    torch.save(output, output_path)

    for prop in PROPERTIES:
        unique, counts = label_tensors[prop].unique(return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info("  %s: %s", prop, dist)

    logger.info("Saved labels to %s (%d samples)", output_path, n_samples)

    # Control-task labels (Hewitt & Liang 2019).  Each property gets a
    # structure-preserving deterministic random label.  Selectivity is then
    # computed downstream as (real_accuracy - control_accuracy).
    if args.control_task:
        control_tensors = {}
        for prop in PROPERTIES:
            if prop not in CONTROL_UNIT:
                logger.warning(
                    "No control-task unit registered for %s — skipping", prop
                )
                continue
            real = label_tensors[prop].numpy()
            if len(np.unique(real)) < 2:
                logger.info("  Skipping control-task for %s (single class)", prop)
                continue
            unit_seq = build_units(
                prop,
                lemma_strs=units["lemma"],
                target_tag_strs=units["target_tag"],
                src1_tag_strs=units["src1_tag"],
                src2_tag_strs=units["src2_tag"],
                stem_strs=units["stem"],
            )
            ctrl = make_control_labels(unit_seq, real, salt=f"{args.control_salt}::{prop}")
            control_tensors[prop] = torch.from_numpy(ctrl).long()
            n_unique_units = len(set(unit_seq))
            logger.info(
                "  control_task[%s]: %d unique %s units -> %d labels",
                prop, n_unique_units, CONTROL_UNIT[prop], len(ctrl),
            )

        control_path = os.path.join(
            args.output_dir, f"{args.split}_{args.run}_control_labels.pt"
        )
        torch.save(
            {
                **control_tensors,
                "label_encodings": LABEL_ENCODINGS,
                "n_samples": n_samples,
                "control_salt": args.control_salt,
                "control_unit": {p: CONTROL_UNIT.get(p) for p in control_tensors},
            },
            control_path,
        )
        logger.info("Saved control-task labels to %s", control_path)

    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
