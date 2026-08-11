#!/usr/bin/env python3
"""Morphome-structure probes: cell clustering and L/NL within non-ar verbs.

Two analyses addressing the structural half of the morphome question, run on
the content-pooled cache at the layers where morphomic information peaks
(enc3, dec1, dec2).

Analysis CELL -- does 1SG.IND cluster with the subjunctive?
  A linear mood probe (IND vs SBJV) is trained on all target cells EXCEPT
  1SG.IND, then applied to held-out-lemma 1SG.IND instances. If the model
  represents the morphomic organization, 1SG.IND instances of L-SHAPED verbs
  should fall on the subjunctive side of the mood boundary more often than
  those of NL-shaped verbs, for which 1SG.IND is an ordinary indicative cell.
  Reported per layer: mood balanced accuracy on non-1SG.IND cells (sanity),
  and mean P(SBJV) / fraction classified SBJV for 1SG.IND instances, split by
  lemma class.

Analysis NONAR -- is L/NL decodable within non-ar verbs?
  L-shaped verbs are all -er/-ir, so conjugation one-directionally predicts
  NL and a probe direction aligned with the model's conjugation encoding
  inherits L/NL signal for free. The transparent control is to restrict the
  probe to non-ar instances, where the shortcut is unavailable: the L/NL
  probe is trained and evaluated (lemma-disjoint folds, balanced accuracy,
  lemma-permutation control) on -er/-ir instances only. Decodability in this
  subset cannot be explained by conjugation-as-proxy.

Output: <output-dir>/<model_type>/<split>_<run>_structure.csv
Exit codes: 0 ok, 1 error, 2 skipped.

Usage:
  run_morphome_structure.py --model-type TYPE --split SPLIT --run RUN
                            [--data-dir DIR] [--pooled-cache-dir DIR]
                            [--output-dir DIR] [--config FILE]
                            [--n-controls N] [--seed N]
  run_morphome_structure.py (-h | --help)

Options:
  --model-type TYPE       Architecture (one of the five MODEL_TYPES).
  --split SPLIT           Data split, e.g. 10L_90NL.
  --run RUN               Run identifier, e.g. 1_2.
  --data-dir DIR          Root data dir [default: FEATURE_INFORMED_DATA].
  --pooled-cache-dir DIR  Pooled cache root [default: data/probing/pooled_cache].
  --output-dir DIR        Output root [default: data/probing/results_structure].
  --config FILE           Probe config [default: probing/config.json].
  --n-controls N          Control label permutations [default: 5].
  --seed N                Fold seed [default: 42].
"""

import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate

from probing import EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.analysis_common import (
    load_labels_and_groups,
    load_layer,
    output_path_or_skip,
    setup_logging,
    write_rows,
)
from probing.extract_labels import _L_CELLS
from probing.run_probes_stemfinal_lnl import (
    build_probe,
    get_src_path,
    load_config,
    shuffle_labels_by_group,
)
from probing.utils.cli import parse, standard_sentinels

logger = setup_logging(__name__)

LAYERS = [("encoder", 3), ("decoder", 1), ("decoder", 2)]
ONE_SG_IND = "V;IND;PRS;1;SG"


def parse_args():
    return parse(__doc__, types=dict(n_controls=int, seed=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS),
                 sentinels=standard_sentinels())


def target_tags(data_dir, split, run):
    """Per-instance target tag string (last ' # ' field of the src line)."""
    with open(get_src_path(data_dir, split, run)) as f:
        return [line.strip().split(" # ")[-1].strip().strip("<>") for line in f]


def grouped_balanced_cv(X, y, groups, config, n_folds, seed):
    """Balanced accuracy of the standard linear probe, lemma-disjoint folds."""
    folds = min(n_folds, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(build_probe("linear", config), X, y, groups=groups,
                         cv=cv, scoring=("balanced_accuracy",), n_jobs=1)
    return float(res["test_balanced_accuracy"].mean())


def main():
    args = parse_args()
    out_path = output_path_or_skip(args.output_dir, args.model_type,
                                   f"{args.split}_{args.run}_structure.csv", logger)

    config = load_config(args.config)
    labels, groups = load_labels_and_groups(args.data_dir, args.split, args.run)
    tags = target_tags(args.data_dir, args.split, args.run)
    y_lsh = np.asarray(labels["l_shaped"])
    y_conj = np.asarray(labels["conjugation"])
    vals, counts = np.unique(y_lsh, return_counts=True)
    l_code = int(vals[np.argmin(counts)])
    is_L = y_lsh == l_code
    y_mood = np.array([0 if "IND" in t.split(";") else 1 for t in tags])  # 0=IND 1=SBJV
    is_1sgind = np.array([t == ONE_SG_IND for t in tags])
    assert set(np.array(tags)[is_1sgind]) <= _L_CELLS
    non_ar = y_conj != 0  # CONJUGATION_MAP: ar=0

    cache = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")
    n_folds = config["probe"]["cv_folds"]
    rng = np.random.RandomState(config["probe"]["random_seed"])
    rows = []
    for layer_type, layer_index in LAYERS:
        X = load_layer(cache, layer_type, layer_index)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

        # ---- CELL: mood probe trained without 1SG.IND, applied to 1SG.IND ----
        mood_true, mood_pred = [], []
        p_sbjv, side_sbjv, cell_is_L = [], [], []
        for tr, te in cv.split(X, y_mood, groups):
            tr_use = tr[~is_1sgind[tr]]
            pipe = build_probe("linear", config)
            pipe.fit(X[tr_use], y_mood[tr_use])
            te_reg = te[~is_1sgind[te]]
            mood_true.append(y_mood[te_reg])
            mood_pred.append(pipe.predict(X[te_reg]))
            te_1sg = te[is_1sgind[te]]
            if len(te_1sg):
                proba = pipe.predict_proba(X[te_1sg])
                sbjv_col = list(pipe.classes_).index(1)
                p_sbjv.append(proba[:, sbjv_col])
                side_sbjv.append(pipe.predict(X[te_1sg]) == 1)
                cell_is_L.append(is_L[te_1sg])
        mood_bal = balanced_accuracy_score(np.concatenate(mood_true), np.concatenate(mood_pred))
        p_sbjv = np.concatenate(p_sbjv)
        side_sbjv = np.concatenate(side_sbjv)
        cell_is_L = np.concatenate(cell_is_L)
        row = dict(layer_type=layer_type, layer_index=layer_index,
                   mood_bal_acc=mood_bal,
                   p_sbjv_1sgind_L=float(p_sbjv[cell_is_L].mean()),
                   p_sbjv_1sgind_NL=float(p_sbjv[~cell_is_L].mean()),
                   frac_sbjv_1sgind_L=float(side_sbjv[cell_is_L].mean()),
                   frac_sbjv_1sgind_NL=float(side_sbjv[~cell_is_L].mean()),
                   n_1sgind_L=int(cell_is_L.sum()), n_1sgind_NL=int((~cell_is_L).sum()))

        # ---- NONAR: L/NL within -er/-ir instances only ----
        Xn, yn, gn = X[non_ar], y_lsh[non_ar], groups[non_ar]
        bal = grouped_balanced_cv(Xn, yn, gn, config, n_folds, args.seed)
        ctrls = []
        for _ in range(args.n_controls):
            y_c = shuffle_labels_by_group(yn, gn, rng)
            if len(np.unique(y_c)) < 2:
                continue
            ctrls.append(grouped_balanced_cv(Xn, y_c, gn, config, n_folds, args.seed))
        ctrl = float(np.mean(ctrls)) if ctrls else float("nan")
        row.update(
            lshaped_nonar=bal,
            lshaped_nonar_control=ctrl,
            lshaped_nonar_selectivity=bal - ctrl,
            n_nonar=int(non_ar.sum()),
            n_nonar_L_lemmas=len(np.unique(gn[yn == l_code])),
            n_nonar_NL_lemmas=len(np.unique(gn[yn != l_code])),
        )
        rows.append(row)
        logger.info("%s_%d | mood=%.3f P(SBJV|1sgind) L=%.3f NL=%.3f | "
                    "lsh-nonar=%.3f ctrl=%.3f sel=%+.3f",
                    layer_type, layer_index, mood_bal,
                    row["p_sbjv_1sgind_L"], row["p_sbjv_1sgind_NL"],
                    bal, ctrl, bal - ctrl)

    write_rows(out_path, rows, logger)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
