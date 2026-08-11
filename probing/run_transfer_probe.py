#!/usr/bin/env python3
"""Cross-subset transfer probe: does ONE L/NL representation span both regimes?

Trains the L-vs-NL probe on instances where the stem alternation is VISIBLE
(stem_final_match = differ) and evaluates on instances of HELD-OUT lemmas
where it is NOT (stem_final_match = same), and vice versa. If the decision
direction learned from alternation-visible instances transfers to
no-alternation instances of unseen lemmas, the model carries a single
morphome representation rather than separate "alternation seen" and "lexeme
class" codes.

Protocol matches the main probes: pooled content vectors from the local
cache, linear probe (scaler + logistic regression), 5-fold lemma-disjoint
StratifiedGroupKFold over the FULL instance set; per fold, the probe is fit
on the train-lemma instances of the SOURCE subset and scored on the
test-lemma instances of the TARGET subset. Balanced accuracy over the pooled
out-of-fold predictions; control = lemma-level label-assignment permutation
averaged over --n-controls draws.

Output: <output-dir>/<model_type>/<split>_<run>_transfer.csv
Exit codes: 0 ok, 1 error, 2 skipped.

Usage:
  run_transfer_probe.py --model-type TYPE --split SPLIT --run RUN
                        [--data-dir DIR] [--pooled-cache-dir DIR]
                        [--output-dir DIR] [--config FILE]
                        [--n-controls N] [--seed N]
  run_transfer_probe.py (-h | --help)

Options:
  --model-type TYPE       Architecture (one of the five MODEL_TYPES).
  --split SPLIT           Data split, e.g. 10L_90NL.
  --run RUN               Run identifier, e.g. 1_2.
  --data-dir DIR          Root data dir [default: FEATURE_INFORMED_DATA].
  --pooled-cache-dir DIR  Pooled cache root [default: data/probing/pooled_cache].
  --output-dir DIR        Output root [default: data/probing/results_transfer].
  --config FILE           Probe config [default: probing/config.json].
  --n-controls N          Control label permutations [default: 5].
  --seed N                Fold seed [default: 42].
"""

import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from probing import EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.analysis_common import (
    LAYERS,
    load_labels_and_groups,
    load_layer,
    output_path_or_skip,
    setup_logging,
    write_rows,
)
from probing.run_probes_stemfinal_lnl import build_probe, load_config, shuffle_labels_by_group
from probing.utils.cli import parse, standard_sentinels

logger = setup_logging(__name__)

DIRECTIONS = (("differ", "same"), ("same", "differ"))


def parse_args():
    return parse(__doc__, types=dict(n_controls=int, seed=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS),
                 sentinels=standard_sentinels())


def transfer_score(X, y, groups, src_mask, tgt_mask, probe, n_folds, seed):
    """Out-of-fold transfer predictions: fit on train-lemmas of the source
    subset, predict test-lemmas of the target subset."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true, y_pred = [], []
    for tr, te in cv.split(X, y, groups):
        tr_idx = tr[src_mask[tr]]
        te_idx = te[tgt_mask[te]]
        if len(te_idx) == 0 or len(np.unique(y[tr_idx])) < 2:
            continue
        est = probe()
        est.fit(X[tr_idx], y[tr_idx])
        y_true.append(y[te_idx])
        y_pred.append(est.predict(X[te_idx]))
    if not y_true:
        return float("nan")
    return balanced_accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))


def main():
    args = parse_args()
    out_path = output_path_or_skip(args.output_dir, args.model_type,
                                   f"{args.split}_{args.run}_transfer.csv", logger)

    config = load_config(args.config)
    labels, groups = load_labels_and_groups(args.data_dir, args.split, args.run)
    y = np.asarray(labels["l_shaped"])
    sfm = np.asarray(labels["stem_final_match"])  # 0 = same, 1 = differ
    masks = {"same": sfm == 0, "differ": sfm == 1}
    rng = np.random.RandomState(config["probe"]["random_seed"])
    n_folds = config["probe"]["cv_folds"]
    cache = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")

    def probe_factory():
        return build_probe("linear", config)

    rows = []
    for layer_type, layer_index in LAYERS:
        X = load_layer(cache, layer_type, layer_index)
        for src, tgt in DIRECTIONS:
            bal = transfer_score(X, y, groups, masks[src], masks[tgt],
                                 probe_factory, n_folds, args.seed)
            ctrls = []
            for _ in range(args.n_controls):
                y_ctrl = shuffle_labels_by_group(y, groups, rng)
                ctrls.append(transfer_score(X, y_ctrl, groups, masks[src], masks[tgt],
                                            probe_factory, n_folds, args.seed))
            ctrl = float(np.nanmean(ctrls))
            rows.append(dict(
                layer_type=layer_type, layer_index=layer_index,
                direction=f"{src}->{tgt}", balanced_accuracy=bal,
                control_balanced_accuracy=ctrl, selectivity=bal - ctrl,
                n_controls=len(ctrls),
            ))
            logger.info("%s_%d | %s->%s: bal=%.4f ctrl=%.4f sel=%+.4f",
                        layer_type, layer_index, src, tgt, bal, ctrl, bal - ctrl)

    write_rows(out_path, rows, logger)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
