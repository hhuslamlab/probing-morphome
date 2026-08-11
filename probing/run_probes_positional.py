#!/usr/bin/env python3
"""Probes on the stem-final-position readout (pool_stemfinal_position.py).

Same protocol as run_probes_stemfinal_lnl.py — linear + MLP probes, 5-fold
lemma-disjoint StratifiedGroupKFold, balanced accuracy, structure-preserving
controls — but the input is the hidden state AT the stem-final consonant
position instead of the content mean. Probes the three main properties plus
l_shaped restricted to the stemfinal_same subset (the position-targeted
version of the dissociation analysis).

Output: <output-dir>/<model_type>/<split>_<run>_positional.csv
Exit codes: 0 ok, 1 error, 2 skipped.

Usage:
  run_probes_positional.py --model-type TYPE --split SPLIT --run RUN
                           [--data-dir DIR] [--pooled-cache-dir DIR]
                           [--output-dir DIR] [--config FILE]
                           [--n-controls N] [--probe-types TYPES]
                           [--suffix NAME]
  run_probes_positional.py (-h | --help)

Options:
  --model-type TYPE       Architecture (one of the five MODEL_TYPES).
  --split SPLIT           Data split, e.g. 10L_90NL.
  --run RUN               Run identifier, e.g. 1_2.
  --data-dir DIR          Root data dir [default: FEATURE_INFORMED_DATA].
  --pooled-cache-dir DIR  Cache with *_stemfinal.npy [default: data/probing/pooled_cache].
  --output-dir DIR        Output root [default: data/probing/results_positional].
  --config FILE           Probe config [default: probing/config.json].
  --n-controls N          Control label permutations [default: 5].
  --probe-types TYPES     Space-separated probe families [default: linear].
                          MLP never exceeded linear on the pooled readout and
                          triples the sweep cost.
  --suffix NAME           Readout to probe: stemfinal (the alternant position)
                          or prealt (the decoder state immediately BEFORE the
                          alternant, which has not yet seen it; decoder layers
                          only) [default: stemfinal].
"""

import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from probing import EXIT_ERROR, EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.analysis_common import (
    LAYERS,
    load_labels_and_groups,
    load_layer,
    output_path_or_skip,
    setup_logging,
    write_rows,
)
from probing.run_probes_stemfinal_lnl import (
    CONTROL_MODE,
    build_probe,
    load_config,
    shuffle_labels_by_group,
)
from probing.utils.cli import parse, standard_sentinels

logger = setup_logging(__name__)

PROPERTIES = ("stem_final_match", "conjugation", "l_shaped")


def parse_args():
    args = parse(__doc__, types=dict(n_controls=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS,
                              suffix=("stemfinal", "prealt")),
                 sentinels=standard_sentinels())
    args.probe_types = args.probe_types.split()
    bad = set(args.probe_types) - {"linear", "mlp"}
    if bad:
        raise SystemExit(f"invalid --probe-types: {sorted(bad)}")
    return args


def probe_cell(X, y, groups, probe_type, config, rng, n_controls, control_mode):
    n_folds = min(config["probe"]["cv_folds"], len(np.unique(groups)))
    seed = config["probe"]["random_seed"]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pipe = build_probe(probe_type, config)
    real = cross_validate(pipe, X, y, groups=groups, cv=cv,
                          scoring=("balanced_accuracy",), n_jobs=1)
    bal = float(real["test_balanced_accuracy"].mean())
    ctrls = []
    for _ in range(n_controls):
        if control_mode == "group":
            y_c = shuffle_labels_by_group(y, groups, rng)
        else:
            y_c = y.copy()
            rng.shuffle(y_c)
        if len(np.unique(y_c)) < 2:
            continue
        cv_c = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        c = cross_validate(build_probe(probe_type, config), X, y_c, groups=groups,
                           cv=cv_c, scoring=("balanced_accuracy",), n_jobs=1)
        ctrls.append(float(c["test_balanced_accuracy"].mean()))
    ctrl = float(np.mean(ctrls)) if ctrls else float("nan")
    return bal, ctrl


def main():
    args = parse_args()
    fname = (f"{args.split}_{args.run}_positional.csv" if args.suffix == "stemfinal"
             else f"{args.split}_{args.run}_positional_{args.suffix}.csv")
    out_path = output_path_or_skip(args.output_dir, args.model_type, fname, logger)

    layers = LAYERS if args.suffix == "stemfinal" else [("decoder", i) for i in range(4)]
    cache = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")
    valid_name = "stemfinal_valid.npy" if args.suffix == "stemfinal" else "prealt_valid.npy"
    valid_path = os.path.join(cache, valid_name)
    if not os.path.exists(valid_path):
        logger.error("No %s readout cache at %s", args.suffix, cache)
        sys.exit(EXIT_ERROR)
    valid = np.load(valid_path)

    config = load_config(args.config)
    labels, groups = load_labels_and_groups(args.data_dir, args.split, args.run)
    sfm = np.asarray(labels["stem_final_match"])
    rng = np.random.RandomState(config["probe"]["random_seed"])

    rows = []
    for layer_type, layer_index in layers:
        X_full = load_layer(cache, layer_type, layer_index, suffix=args.suffix)
        cells = [(prop, np.ones(len(valid), bool), "all") for prop in PROPERTIES]
        cells.append(("l_shaped", sfm == 0, "stemfinal_same"))
        for prop, sel, subset in cells:
            m = valid & sel
            X = X_full[m]
            y = np.asarray(labels[prop])[m]
            g = groups[m]
            for probe_type in args.probe_types:
                mode = CONTROL_MODE.get(prop, "sample")
                bal, ctrl = probe_cell(X, y, g, probe_type, config, rng,
                                       args.n_controls, mode)
                rows.append(dict(
                    layer_type=layer_type, layer_index=layer_index, subset=subset,
                    property=prop, probe_type=probe_type, balanced_accuracy=bal,
                    control_balanced_accuracy=ctrl, selectivity=bal - ctrl,
                    n_samples=int(m.sum()),
                ))
                logger.info("%s_%d | %s | %s | %s: bal=%.4f ctrl=%.4f sel=%+.4f",
                            layer_type, layer_index, subset, prop, probe_type,
                            bal, ctrl, bal - ctrl)

    write_rows(out_path, rows, logger)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
