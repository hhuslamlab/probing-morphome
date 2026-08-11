"""L vs NL probing within the shared-stem-final subset.

Probes whether l_shaped is still decodable on the stemfinal_same subset, where
the surface alternation cue is absent, alongside the stemfinal_differ and all
subsets. The metric is balanced accuracy because L vs NL is heavily
class-imbalanced (L is ~2.4% of stemfinal_same samples). Folds are
lemma-disjoint so a positive result reflects generalization to unseen lemmas,
not lemma memorization. Caveat for 10L_90NL: its test set has only ~7 distinct
L-lemmas, so treat a positive result there as suggestive (see n_L_lemmas).

Usage:
  run_probes_lnl_within_stemfinal.py --model-type TYPE --split SPLIT --run RUN
                                     [--representations-dir DIR] [--data-dir DIR]
                                     [--output-dir DIR] [--config FILE]
                                     [--pool-positions POS] [--n-controls N]
                                     [--n-jobs N]
  run_probes_lnl_within_stemfinal.py (-h | --help)

Options:
  --model-type TYPE          Architecture (one of the five MODEL_TYPES).
  --split SPLIT              Data split, e.g. 10L_90NL.
  --run RUN                  Run identifier, e.g. 1_2.
  --representations-dir DIR  Directory with extracted representations
                             [default: data/probing/representations].
  --data-dir DIR             Root data directory containing split/test/runN/
                             folders; the raw test data lives in the
                             feature_informed repo [default: FEATURE_INFORMED_DATA].
  --output-dir DIR           Output directory for probe results
                             [default: data/probing/results_lnl_within_stemfinal].
  --config FILE              Probe config file [default: probing/config.json].
  --pool-positions POS       Pooling positions; 'content' means mean over
                             chars, no tag leakage [default: content].
  --n-controls N             Lemma-level label permutations to average the
                             control over [default: 5].
  --n-jobs N                 Parallel workers for cross_validate (folds run
                             concurrently; scores are identical to 1)
                             [default: 1].
"""

import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from probing import EXIT_ERROR, EXIT_SKIPPED, EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.extract_labels import LSHAPED_MAP, STEMFINAL_MATCH_MAP
from probing.utils.cli import parse, standard_sentinels
from probing.run_probes_stemfinal_lnl import (
    build_lemma_groups,
    build_probe,
    build_property_labels,
    get_src_path,
    get_tgt_path,
    load_config,
    shuffle_labels_by_group,
)
from probing.utils.content_mask import load_pool_mask, pool_reps

# Subsets to probe, by stem_final_match value (None = no filter, the "all" ref).
SUBSETS = (
    ("stemfinal_same", STEMFINAL_MATCH_MAP["same"]),
    ("stemfinal_differ", STEMFINAL_MATCH_MAP["differ"]),
    ("all", None),
)


def parse_args():
    return parse(__doc__,
                 types=dict(n_controls=int, n_jobs=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS,
                              pool_positions=("content", "all", "last")),
                 sentinels=standard_sentinels())


def subset_stats(y, groups):
    """Return (n, n_L, n_NL, n_L_lemmas, n_NL_lemmas, majority_baseline)."""
    n_L = int((y == LSHAPED_MAP["L"]).sum())
    n_NL = int((y == LSHAPED_MAP["NL"]).sum())
    g_L = int(len(np.unique(groups[y == LSHAPED_MAP["L"]])))
    g_NL = int(len(np.unique(groups[y == LSHAPED_MAP["NL"]])))
    majority = max(n_L, n_NL) / len(y) if len(y) else float("nan")
    return len(y), n_L, n_NL, g_L, g_NL, majority


def probe_subset(X, y, groups, subset_name, layer_type, layer_index, config, rng, n_controls=5, n_jobs=1):
    """Probe l_shaped on one subset; return list of result-row dicts (the control
    permutes labels at the lemma level; per-sample shuffles are unlearnable
    under lemma-disjoint folds)."""
    n, n_L, n_NL, g_L, g_NL, majority = subset_stats(y, groups)
    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"WARNING:   {layer_type}_{layer_index} | {subset_name}: skipping (only {n_classes} class in {n} samples)", file=sys.stderr)
        return []

    # StratifiedGroupKFold needs each class represented; the rarer class's lemma
    # count (min(g_L, g_NL)) caps usable folds.
    n_folds = min(config["probe"]["cv_folds"], min(g_L, g_NL))
    if n_folds < 2:
        print(f"WARNING:   {layer_type}_{layer_index} | {subset_name}: skipping (n_folds<2: L lemmas={g_L}, NL lemmas={g_NL})", file=sys.stderr)
        return []

    seed = config["probe"]["random_seed"]
    scoring = ("balanced_accuracy", "accuracy")
    results = []
    for probe_type in ("linear", "mlp"):
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        real = cross_validate(
            build_probe(probe_type, config), X, y, groups=groups, cv=cv, scoring=scoring,
            n_jobs=n_jobs,
        )
        bal = real["test_balanced_accuracy"]
        raw = real["test_accuracy"]

        # Control: permute the lemma-to-label assignment (structure-preserving),
        # same protocol, so expected balanced acc ~0.5. Averaged over
        # n_controls permutations; degenerate draws (one class) are skipped.
        ctrl_bals = []
        for _ in range(n_controls):
            y_shuf = shuffle_labels_by_group(y, groups, rng)
            if len(np.unique(y_shuf)) < 2:
                continue
            cv_ctrl = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            ctrl = cross_validate(
                build_probe(probe_type, config), X, y_shuf, groups=groups, cv=cv_ctrl,
                scoring=("balanced_accuracy",), n_jobs=n_jobs,
            )
            ctrl_bals.append(ctrl["test_balanced_accuracy"].mean())
        ctrl_bal = float(np.mean(ctrl_bals))
        ctrl_bal_std = float(np.std(ctrl_bals))

        row = {
            "layer_type": layer_type,
            "layer_index": layer_index,
            "subset": subset_name,
            "probe_type": probe_type,
            "balanced_accuracy": bal.mean(),
            "balanced_std": bal.std(),
            "control_balanced_accuracy": ctrl_bal,
            "control_balanced_std": ctrl_bal_std,
            "selectivity": bal.mean() - ctrl_bal,
            "raw_accuracy": raw.mean(),
            "majority_baseline": majority,
            "n_folds": n_folds,
            "n_controls": len(ctrl_bals),
            "n_samples": n,
            "n_L": n_L,
            "n_NL": n_NL,
            "n_L_lemmas": g_L,
            "n_NL_lemmas": g_NL,
        }
        results.append(row)
    return results


CSV_COLUMNS = (
    "layer_type", "layer_index", "subset", "probe_type",
    "balanced_accuracy", "balanced_std", "control_balanced_accuracy",
    "control_balanced_std", "selectivity",
    "raw_accuracy", "majority_baseline", "n_folds", "n_controls",
    "n_samples", "n_L", "n_NL", "n_L_lemmas", "n_NL_lemmas",
)


if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.join(args.output_dir, args.model_type)
    # Non-default protocol flags land in the filename so a rerun with different
    # settings can never be silently skipped against an existing result file.
    variant = "" if args.pool_positions == "content" else f".poolpositions-{args.pool_positions}"
    if args.n_controls != 5:
        variant += f".ncontrols-{args.n_controls}"
    output_path = os.path.join(
        output_dir, f"{args.split}_{args.run}_lnl_within_stemfinal{variant}.csv"
    )
    if os.path.exists(output_path):
        print(f"Skipping {args.model_type}/{args.split}_{args.run} -- results already exist")
        sys.exit(EXIT_SKIPPED)

    if not os.path.exists(args.config):
        sys.exit(f"Config not found: {args.config}")
    config = load_config(args.config)

    reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        sys.exit(f"Representations not found: {reps_dir}")
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    n_encoder = metadata["n_encoder_layers"]
    n_decoder = metadata["n_decoder_layers"]
    n_samples = metadata["n_samples"]

    src_path = get_src_path(args.data_dir, args.split, args.run)
    tgt_path = get_tgt_path(args.data_dir, args.split, args.run)
    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            sys.exit(f"Data file not found: {path}")

    labels = build_property_labels(src_path, tgt_path, args.data_dir)
    stem_final_match = labels["stem_final_match"]
    l_shaped = labels["l_shaped"]
    for name, y in (("stem_final_match", stem_final_match), ("l_shaped", l_shaped)):
        if len(y) != n_samples:
            sys.exit(f"Sample count mismatch for {name}: data={len(y)} reps={n_samples}")

    groups = build_lemma_groups(args.data_dir, args.split, args.run)

    # Log the subset composition up front (this is where the 10L_90NL caveat shows).
    for subset_name, sf_val in SUBSETS:
        m = np.ones(n_samples, dtype=bool) if sf_val is None else (stem_final_match == sf_val)
        n, n_L, n_NL, g_L, g_NL, base = subset_stats(l_shaped[m], groups[m])

    layers = [("encoder", i) for i in range(n_encoder)] + [("decoder", i) for i in range(n_decoder)]
    rng = np.random.RandomState(config["probe"]["random_seed"])
    all_results = []
    for layer_type, layer_index in layers:
        rep_path = os.path.join(reps_dir, f"{layer_type}_layer_{layer_index}.pt")
        if not os.path.exists(rep_path):
            print(f"WARNING: Skipping {layer_type}_{layer_index}: missing rep file", file=sys.stderr)
            continue
        reps = torch.load(rep_path, weights_only=False)
        mask = load_pool_mask(reps_dir, layer_type, layer_index, args.pool_positions)
        if mask is None:
            print(f"WARNING: Skipping {layer_type}_{layer_index}: missing mask file", file=sys.stderr)
            continue
        X_full = pool_reps(reps, mask, args.pool_positions).numpy()

        for subset_name, sf_val in SUBSETS:
            m = np.ones(n_samples, dtype=bool) if sf_val is None else (stem_final_match == sf_val)
            rows = probe_subset(
                X_full[m], l_shaped[m], groups[m], subset_name,
                layer_type, layer_index, config, rng, n_controls=args.n_controls,
                n_jobs=args.n_jobs,
            )
            all_results.extend(rows)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for row in all_results:
            f.write(",".join(str(row[c]) for c in CSV_COLUMNS) + "\n")
    sys.exit(EXIT_SUCCESS)
