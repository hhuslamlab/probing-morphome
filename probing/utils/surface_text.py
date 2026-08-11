"""Parse .src/.tgt surface strings into text views for surface-form baselines.

A .src line is ``form1 <tag1> # form2 <tag2> # <target_tag>`` with space-
separated IPA phoneme tokens; tags are single ``<...>`` tokens. The views built
here are the surface analogs of the probing readouts:

  - src-content:   the two source forms with tags and ``#`` separators dropped
    -- exactly the tokens content pooling keeps on the ENCODER side. A literal
    ``|`` boundary token is kept between the forms so n-grams never span forms.
  - tgt-content:   the target form -- the DECODER-side content tokens.
  - all-content:   both source forms plus the target form. Decoder states
    attend to the encoder, so this is the fair surface analog of decoder
    probes, and the only view with full information for the relational
    stem_final_match label (which compares all three forms).
  - src-with-tags: the raw src line including tag tokens (analog of tag
    pooling; opt-in, leaks the morphological tags).
"""

FORM_BOUNDARY = "|"


def parse_src_line(src_line):
    """Split one .src line into (forms, tags, target_tag).

    forms: list of phoneme-token lists, one per source form.
    tags: the ``<...>`` tag token of each source form ('' if absent).
    target_tag: the bare final ``<target_tag>`` segment.
    """
    segments = [seg.strip() for seg in src_line.strip().split(" # ")]
    if len(segments) < 2:
        raise ValueError(f"Malformed src line (no '#' separators): {src_line!r}")
    *form_segments, target_tag = segments
    forms, tags = [], []
    for seg in form_segments:
        toks = seg.split()
        forms.append([t for t in toks if not t.startswith("<")])
        tag_toks = [t for t in toks if t.startswith("<")]
        tags.append(tag_toks[0] if tag_toks else "")
    return forms, tags, target_tag


def build_texts(src_lines, tgt_lines, with_tags=False):
    """Build all text views over parallel src/tgt lines.

    Returns a dict mapping view name -> list of space-joined token strings,
    row-aligned with the input lines (and hence with the probe labels).
    """
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt")
    views = {"src-content": [], "tgt-content": [], "all-content": []}
    if with_tags:
        views["src-with-tags"] = []
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        forms, _tags, _target_tag = parse_src_line(src_line)
        src_text = f" {FORM_BOUNDARY} ".join(" ".join(f) for f in forms)
        tgt_text = tgt_line.strip()
        views["src-content"].append(src_text)
        views["tgt-content"].append(tgt_text)
        views["all-content"].append(f"{src_text} {FORM_BOUNDARY} {tgt_text}")
        if with_tags:
            views["src-with-tags"].append(src_line.strip())
    return views
