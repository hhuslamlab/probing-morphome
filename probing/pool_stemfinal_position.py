"""Stem-final-position readout: one vector per instance at the alternation site.

Encoder layers average the two source forms' stem-final positions; decoder
layers take the target's, plus a pre-alternant readout (the state one content
position earlier, which under teacher forcing has not yet seen the alternant).
Positions come from aligning form tokens with the content mask; rows failing
the per-row count check are marked invalid. Exit codes: 0 ok, 1 error, 2 skipped.

Usage:
  pool_stemfinal_position.py --model-type TYPE --split SPLIT --run RUN --representations-dir DIR
                             [--data-dir DIR] [--cache-dir DIR]
  pool_stemfinal_position.py (-h | --help)

Options:
  --model-type TYPE          Architecture (one of the five MODEL_TYPES).
  --split SPLIT              Data split, e.g. 10L_90NL.
  --run RUN                  Run identifier, e.g. 1_2.
  --representations-dir DIR  Root of the extracted representations (drive).
  --data-dir DIR             Root data dir [default: FEATURE_INFORMED_DATA].
  --cache-dir DIR            Cache root [default: data/probing/pooled_cache].
"""

import os

import numpy as np
import torch

from probing import EXIT_SKIPPED, MODEL_TYPES, SPLITS
from probing.analysis_common import LAYERS
from probing.extract_labels import _get_stem
from probing.run_probes_stemfinal_lnl import get_src_path, get_tgt_path
from probing.utils.cli import parse, standard_sentinels

VOWELS = set("aeiou")
STRESS = {"ˈ", "ˌ"}


def parse_args():
    return parse(__doc__, choices=dict(model_type=MODEL_TYPES, split=SPLITS),
                 sentinels=standard_sentinels())


def form_tokens(segment):
    """Token list of one src segment's form (chars + stress marks, no tags)."""
    toks = []
    for tok in segment.split():
        if tok.isupper() or tok.isdigit():
            break
        toks.append(tok)
    return toks


def stem_final_token_index(toks):
    """Index into toks of the stem-final consonant (falls back to the last stem token)."""
    joined = "".join(toks)
    stem = _get_stem(joined)
    if stem is None:
        stem = joined
    # walk tokens until the joined length covers the stem
    covered, stem_end = 0, len(toks) - 1
    for i, t in enumerate(toks):
        covered += len(t)
        if covered >= len(stem):
            stem_end = i
            break
    for i in range(stem_end, -1, -1):
        t = toks[i]
        if t in STRESS:
            continue
        if not (set(t) <= VOWELS):
            return i
    return stem_end


if __name__ == "__main__":
    args = parse_args()


    reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    cache_dir = os.path.join(args.cache_dir, args.model_type, f"{args.split}_{args.run}")
    if not os.path.exists(os.path.join(reps_dir, "metadata.json")):
        raise FileNotFoundError(f"No representations at {reps_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    targets = [os.path.join(cache_dir, f"{lt}_layer_{li}_stemfinal.npy") for lt, li in LAYERS]
    targets += [os.path.join(cache_dir, f"decoder_layer_{li}_prealt.npy") for li in range(4)]
    valid_path = os.path.join(cache_dir, "stemfinal_valid.npy")
    prealt_valid_path = os.path.join(cache_dir, "prealt_valid.npy")
    if (all(os.path.exists(t) for t in targets) and os.path.exists(valid_path)
            and os.path.exists(prealt_valid_path)):
        print(f"Skipping {args.model_type}/{args.split}_{args.run} -- all stemfinal/prealt readouts cached")
        raise SystemExit(EXIT_SKIPPED)

    with open(get_src_path(args.data_dir, args.split, args.run)) as f:
        src_lines = [l.strip() for l in f]
    with open(get_tgt_path(args.data_dir, args.split, args.run)) as f:
        tgt_lines = [l.strip() for l in f]

    # Per-instance token structure and stem-final indices.
    enc_counts, enc_sf1, enc_sf2 = [], [], []
    dec_counts, dec_sf = [], []
    for src, tgt in zip(src_lines, tgt_lines):
        *segs, _ = src.split(" # ")
        f1, f2 = form_tokens(segs[0]), form_tokens(segs[1])
        enc_counts.append(len(f1) + len(f2))
        enc_sf1.append(stem_final_token_index(f1))
        enc_sf2.append(len(f1) + stem_final_token_index(f2))
        tt = tgt.split()
        dec_counts.append(len(tt))
        dec_sf.append(stem_final_token_index(tt))
    enc_counts = np.array(enc_counts)
    dec_counts = np.array(dec_counts)

    dec_sf_arr = np.array(dec_sf)
    # Resume-safe validity: when a pass re-reads only some layers (e.g. adding
    # prealt to a cache that already has stemfinal), start from the recorded
    # validity instead of rebuilding it from the re-read layers alone.
    valid = np.load(valid_path) if os.path.exists(valid_path) else None
    for layer_type, layer_index in LAYERS:
        base = f"{layer_type}_layer_{layer_index}"
        out = os.path.join(cache_dir, f"{base}_stemfinal.npy")
        out_pre = os.path.join(cache_dir, f"{base}_prealt.npy")
        need_sf = not os.path.exists(out)
        need_pre = layer_type == "decoder" and not os.path.exists(out_pre)
        if not need_sf and not need_pre:
            continue
        reps = torch.load(os.path.join(reps_dir, f"{base}.pt"), weights_only=False)
        mask = torch.load(os.path.join(reps_dir, f"{base}_content_mask.pt"), weights_only=False)
        mask = mask.bool().numpy()
        n = reps.shape[0]
        counts = mask.sum(axis=1)
        expect = enc_counts if layer_type == "encoder" else dec_counts
        row_ok = counts == expect
        if valid is None:
            valid = row_ok.copy()
        else:
            valid &= row_ok
        pooled = np.zeros((n, reps.shape[2]), dtype=np.float32)
        pooled_pre = np.zeros((n, reps.shape[2]), dtype=np.float32) if need_pre else None
        for r in range(n):
            if not row_ok[r]:
                continue
            pos = np.flatnonzero(mask[r])
            if layer_type == "encoder":
                v = (reps[r, pos[enc_sf1[r]]] + reps[r, pos[enc_sf2[r]]]) / 2.0
            else:
                v = reps[r, pos[dec_sf[r]]]
                if need_pre and dec_sf[r] > 0:
                    vp = reps[r, pos[dec_sf[r] - 1]]
                    pooled_pre[r] = vp.numpy() if hasattr(vp, "numpy") else vp
            pooled[r] = v.numpy() if hasattr(v, "numpy") else v
        if need_sf:
            tmp = f"{out}.{os.getpid()}.tmp.npy"
            np.save(tmp, pooled)
            os.replace(tmp, out)
        if need_pre:
            tmp = f"{out_pre}.{os.getpid()}.tmp.npy"
            np.save(tmp, pooled_pre)
            os.replace(tmp, out_pre)
        del reps, mask

    np.save(valid_path, valid)
    np.save(prealt_valid_path, valid & (dec_sf_arr > 0))
    frac = float(valid.mean())
    if frac < 0.99:
        raise ValueError("Alignment below 99% -- investigate before probing")
