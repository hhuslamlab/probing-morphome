#!/usr/bin/env python3
"""Nonce-verb (wug) probing: do models internally classify novel alternating
stems as L-shaped?

Pipeline per (architecture, run):
  1. extract representations for the 120 Nevins et al. nonce re-inflection
     items (two source forms exhibiting a novel stem-final alternation +
     target tag), teacher-forcing the L-shape-consistent EXPECTED form
     (identical across models, from lshaped_form_msd_mapping.json);
  2. content-mean-pool each layer (same readout as the main probes);
  3. train the l_shaped linear probe on the REAL test set's pooled cache
     (full 43,560 instances) and apply it to the 120 nonce vectors;
  4. record, per layer, the fraction of nonce items classified L-shaped and
     the mean P(L) -- the representational analogue of the behavioral wug
     generalization rate.

Output: <output-dir>/<model_type>/<split>_<run>_nonce.csv
Exit codes: 0 ok, 1 error, 2 skipped.

Usage:
  run_nonce_probe.py --prepare [--nonce-dir DIR]
  run_nonce_probe.py --model-type TYPE --split SPLIT --run RUN
                     [--data-dir DIR] [--pooled-cache-dir DIR]
                     [--nonce-dir DIR] [--reps-dir DIR] [--output-dir DIR]
                     [--config FILE]
  run_nonce_probe.py (-h | --help)

Options:
  --prepare               Build the shared expected-form nonce .tgt and exit.
  --model-type TYPE       Architecture (one of the five MODEL_TYPES).
  --split SPLIT           Data split [default: 10L_90NL].
  --run RUN               Run identifier, e.g. 1_2.
  --data-dir DIR          Root data dir [default: FEATURE_INFORMED_DATA].
  --pooled-cache-dir DIR  Real-corpus pooled cache [default: data/probing/pooled_cache].
  --nonce-dir DIR         Nonce working dir [default: data/probing/nonce].
  --reps-dir DIR          Nonce representations [default: data/probing/representations_nonce].
  --output-dir DIR        Output root [default: data/probing/results_nonce].
  --config FILE           Probe config [default: probing/config.json].
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from probing import EXIT_ERROR, EXIT_SUCCESS, FEATURE_INFORMED_ROOT, MODEL_TYPES, SPLITS
from probing.analysis_common import (
    LAYERS,
    load_labels_and_groups,
    load_layer,
    output_path_or_skip,
    setup_logging,
    write_rows,
)
from probing.run_probes_stemfinal_lnl import build_probe, load_config
from probing.utils.cli import parse, standard_sentinels
from probing.utils.content_mask import load_pool_mask, pool_reps

logger = setup_logging(__name__)

NEVINS = os.path.join(FEATURE_INFORMED_ROOT, "data", "nevins_data")


def parse_args():
    # cli.parse skips None values in the choices check, so the --prepare
    # invocation (no --model-type) validates cleanly.
    return parse(__doc__, choices=dict(model_type=MODEL_TYPES, split=SPLITS),
                 sentinels=standard_sentinels())


def _segment_form(seg):
    toks = []
    for t in seg.split():
        if t.startswith("<") or t.isupper() or t.isdigit():
            break
        toks.append(t)
    return toks


def prepare_tgt(nonce_dir):
    """Derive the expected-form .tgt for the 120 nonce items."""
    with open(os.path.join(NEVINS, "nevins_test.src")) as f:
        src_lines = [l.strip() for l in f]
    forms = json.load(open(os.path.join(NEVINS, "lshaped_form_msd_mapping.json")))
    # form -> (verb, msd) reverse index
    rev = {}
    for verb, cells in forms.items():
        for msd, form in cells.items():
            rev.setdefault(form, (verb, msd))
    out = []
    for line in src_lines:
        *segs, tgt_tag = line.split(" # ")
        tgt_tag = tgt_tag.strip().strip("<>")
        verb = None
        for seg in segs:
            f = "".join(_segment_form(seg))
            if f in rev:
                verb = rev[f][0]
                break
        if verb is None:
            raise SystemExit(f"Cannot identify nonce verb in: {line}")
        expected = forms[verb][tgt_tag]
        out.append(" ".join(expected))
    os.makedirs(nonce_dir, exist_ok=True)
    path = os.path.join(nonce_dir, "nonce_test.tgt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    logger.info("Wrote %s (%d items)", path, len(out))
    return path


def extract(args, tgt_path):
    """Run the architecture's extractor on the nonce items."""
    arch, split, run = args.model_type, args.split, args.run
    out_root = args.reps_dir
    done = os.path.join(out_root, arch, f"{split}_{run}", "encoder_layer_0.pt")
    if os.path.exists(done):
        return os.path.dirname(done)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    src_plain = os.path.join(NEVINS, "nevins_test.src")
    src_delim = os.path.join(NEVINS, "nevins_test_delimited.src")
    if arch == "vanilla":
        cmd = [sys.executable, "-u", "-m", "probing.extract_representations_vanilla",
               "--model", f"{split}_{run}", "--checkpoint-name", "checkpoint_best.pt",
               "--test-src", src_plain, "--test-tgt", tgt_path,
               "--min-token-acc", "0.0", "--num-threads", "2",
               "--output-dir", out_root]
    elif arch == "character_separated":
        ck = os.path.join(FEATURE_INFORMED_ROOT, "checkpoints/char_sep/seperate_char_checkpoints",
                          f"{split}_{run}-models",
                          "checkpoint_last.pt" if run == "1_1" else "checkpoint_best.pt")
        dbin = os.path.join(FEATURE_INFORMED_ROOT, "data/char_sep_databin_aligned", f"{split}_{run}")
        cmd = [sys.executable, "-u", "-m", "probing.extract_representations_char_sep",
               "--model-type", "character_separated", "--checkpoint", ck,
               "--data-bin", dbin, "--test-src", src_delim, "--test-tgt", tgt_path,
               "--output-dir", out_root]
    else:
        # in-house architectures: shadow data dir with nonce test files
        run_num = run.split("_")[0]
        shadow = os.path.join(args.nonce_dir, f"shadow_{arch}_{run}")
        base = os.path.join(shadow, split)
        for subset in ("train", "dev"):
            d = os.path.join(base, subset, f"run{run_num}")
            os.makedirs(d, exist_ok=True)
            for ext in ("src", "tgt"):
                link = os.path.join(d, f"{subset}.{split}_{run}.{ext}")
                real = os.path.join(args.data_dir, split, subset, f"run{run_num}",
                                    f"{subset}.{split}_{run}.{ext}")
                if not os.path.exists(link):
                    os.symlink(real, link)
        d = os.path.join(base, "test", f"run{run_num}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(src_plain, os.path.join(d, f"test.{split}_{run}.src"))
        shutil.copy(tgt_path, os.path.join(d, f"test.{split}_{run}.tgt"))
        if arch == "independent_feature":
            ck = os.path.join(FEATURE_INFORMED_ROOT, "checkpoints/feature_onehot/independentfeature_fixed",
                              f"{split}_{run}.nll_0.0000.epoch_103")
        else:
            ck = os.path.join(FEATURE_INFORMED_ROOT, "checkpoints", arch, f"{split}_{run}")
        cmd = [sys.executable, "-u", "-m", "probing.extract_representations",
               "--model-type", arch, "--split", split, "--run", run,
               "--checkpoint", ck, "--data-dir", shadow, "--output-dir", out_root]
    logger.info("Extracting nonce reps: %s", " ".join(cmd[-8:]))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode not in (0, 2):
        logger.error("Extraction failed (%d):\n%s", r.returncode, r.stdout[-2000:] + r.stderr[-2000:])
        sys.exit(EXIT_ERROR)
    return os.path.join(out_root, arch, f"{split}_{run}")


def main():
    args = parse_args()
    tgt_path = os.path.join(args.nonce_dir, "nonce_test.tgt")
    if args.prepare:
        prepare_tgt(args.nonce_dir)
        sys.exit(EXIT_SUCCESS)
    if not os.path.exists(tgt_path):
        prepare_tgt(args.nonce_dir)
    if not args.model_type or not args.run:
        logger.error("--model-type and --run required (or --prepare)")
        sys.exit(EXIT_ERROR)

    out_path = output_path_or_skip(args.output_dir, args.model_type,
                                   f"{args.split}_{args.run}_nonce.csv", logger)

    nonce_reps = extract(args, tgt_path)

    config = load_config(args.config)
    labels, _ = load_labels_and_groups(args.data_dir, args.split, args.run)
    y_real = np.asarray(labels["l_shaped"])
    # Which label code is L? The minority class (4,620 of 43,560).
    vals, counts = np.unique(y_real, return_counts=True)
    l_code = int(vals[np.argmin(counts)])

    cache = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")
    rows = []
    for layer_type, layer_index in LAYERS:
        X_real = load_layer(cache, layer_type, layer_index)
        reps = torch.load(os.path.join(nonce_reps, f"{layer_type}_layer_{layer_index}.pt"),
                          weights_only=False)
        mask = load_pool_mask(nonce_reps, layer_type, layer_index, "content", logger=logger)
        X_nonce = pool_reps(reps, mask, "content").numpy()
        pipe = build_probe("linear", config)
        pipe.fit(X_real, y_real)
        proba = pipe.predict_proba(X_nonce)
        l_col = list(pipe.classes_).index(l_code)
        p_l = proba[:, l_col]
        frac_l = float((pipe.predict(X_nonce) == l_code).mean())
        rows.append(dict(layer_type=layer_type, layer_index=layer_index,
                         frac_classified_L=frac_l, mean_p_L=float(p_l.mean()),
                         n_items=len(p_l)))
        logger.info("%s_%d: frac_L=%.3f mean_P(L)=%.3f",
                    layer_type, layer_index, frac_l, float(p_l.mean()))
        del reps, X_real

    write_rows(out_path, rows, logger)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
