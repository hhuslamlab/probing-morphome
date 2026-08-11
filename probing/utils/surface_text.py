"""Parse .src/.tgt surface strings into text views for surface-form baselines.

Views mirror the probing readouts: src-content (source forms with tags and
separators dropped), tgt-content (target form), all-content (all three forms;
the fair analog of decoder probes and the only view complete for the
relational stem_final_match label), and opt-in src-with-tags (leaks the tags).
A literal '|' boundary token keeps n-grams from spanning forms.
"""

FORM_BOUNDARY = "|"


def parse_src_line(src_line):
    """Split one .src line into (forms, tags, target_tag)."""
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
    """Build all text views over parallel src/tgt lines (row-aligned with the probe labels)."""
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
