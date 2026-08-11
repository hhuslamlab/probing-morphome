"""L vs NL probing *within* the shared-stem-final subset.

Research question:
The L-shape morphome surfaces as an alternation of the stem-final consonant
across the three forms of an inflection instance. When that alternation IS
present (``stem_final_match == differ``) it is a trivial surface cue for
L-membership. The interesting test removes the cue:

    Restricted to instances where the stem-final consonant is SHARED across the
    three forms (``stem_final_match == same``, no alternation surfacing), can a
    probe STILL classify which forms belong to L-shaped vs NL-shaped lemmas?

A positive answer means the network carries an *abstract* L/NL representation
that does not depend on the visible alternation; a null answer means its L/NL
signal is essentially the surface alternation itself.

Protocol:
For each layer we content mean-pool (no tag leakage, as in
run_probes_stemfinal_lnl.py) and probe ``l_shaped`` on three subsets:
``stemfinal_same`` (the question), ``stemfinal_differ`` (the cued contrast,
should be easy), and ``all`` (reference). Folds are lemma-disjoint
(StratifiedGroupKFold) so a positive result reflects generalization to UNSEEN
lemmas, not a memorized lemma->class lookup.

Metric is **balanced accuracy** (macro recall), NOT raw accuracy: in the
``stemfinal_same`` subset L is ~2.4% of samples, so a majority-class predictor
scores ~0.975 on raw accuracy while balanced accuracy stays at 0.5. The
selectivity control permutes the lemma->L/NL ASSIGNMENT (group-level,
structure-preserving; averaged over --n-controls permutations) — a per-sample
label shuffle would be trivially unlearnable under lemma-disjoint folds and
give a vacuous control.

CAVEAT (10L_90NL): the 10L_90NL test set contains only ~7 distinct L-lemmas, so
the lemma-disjoint generalization here rests on very few groups -- treat a
positive result as suggestive, not conclusive (see n_L_lemmas in the output).

Reuses the label / lemma-group / pooling helpers from run_probes_stemfinal_lnl.py
and extract_labels.py.

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

import logging
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

logger = logging.getLogger(__name__)

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
    """Probe l_shaped on one subset; return list of result-row dicts.

    Balanced accuracy via lemma-disjoint StratifiedGroupKFold, plus a control
    for selectivity. l_shaped is lemma-level, so the control permutes the
    lemma->label ASSIGNMENT (shuffle_labels_by_group) rather than shuffling
    per-sample — a per-sample shuffle is trivially unlearnable under lemma-
    disjoint folds and would make selectivity vacuous. The control is averaged
    over n_controls permutations. Raw accuracy reported alongside for context.
    """
    n, n_L, n_NL, g_L, g_NL, majority = subset_stats(y, groups)
    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.warning(
            "  %s_%d | %s: skipping (only %d class in %d samples)",
            layer_type, layer_index, subset_name, n_classes, n,
        )
        return []

    # StratifiedGroupKFold needs each class represented; the rarer class's lemma
    # count (min(g_L, g_NL)) caps usable folds.
    n_folds = min(config["probe"]["cv_folds"], min(g_L, g_NL))
    if n_folds < 2:
        logger.warning(
            "  %s_%d | %s: skipping (n_folds<2: L lemmas=%d, NL lemmas=%d)",
            layer_type, layer_index, subset_name, g_L, g_NL,
        )
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
        logger.info(
            "  %s_%d | %-16s | %-6s: bal_acc=%.4f (+/-%.4f) ctrl=%.4f sel=%+.4f "
            "[raw=%.4f base=%.4f n=%d L=%d/%dlem folds=%d]",
            layer_type, layer_index, subset_name, probe_type,
            bal.mean(), bal.std(), ctrl_bal, bal.mean() - ctrl_bal,
            raw.mean(), majority, n, n_L, g_L, n_folds,
        )
    return results


CSV_COLUMNS = (
    "layer_type", "layer_index", "subset", "probe_type",
    "balanced_accuracy", "balanced_std", "control_balanced_accuracy",
    "control_balanced_std", "selectivity",
    "raw_accuracy", "majority_baseline", "n_folds", "n_controls",
    "n_samples", "n_L", "n_NL", "n_L_lemmas", "n_NL_lemmas",
)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    output_dir = os.path.join(args.output_dir, args.model_type)
    # Non-default protocol flags land in the filename so a rerun with different
    # settings can never be silently skipped against an existing result file.
    variant = "" if args.pool_positions == "content" else f".poolpositions-{args.pool_positions}"
    if args.n_controls != 5:
        variant += f".ncontrols-{args.n_controls}"
    output_path = os.path.join(
        output_dir, f"{args.split}_{args.run}_lnl_within_stemfinal{variant}.csv"
    )
    logger.info("Output file: %s", output_path)
    if os.path.exists(output_path):
        logger.info("Skipping %s/%s_%s -- results already exist",
                    args.model_type, args.split, args.run)
        sys.exit(EXIT_SKIPPED)

    if not os.path.exists(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(EXIT_ERROR)
    config = load_config(args.config)

    reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error("Representations not found: %s", reps_dir)
        sys.exit(EXIT_ERROR)
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    n_encoder = metadata["n_encoder_layers"]
    n_decoder = metadata["n_decoder_layers"]
    n_samples = metadata["n_samples"]
    logger.info("Loaded metadata: %d encoder + %d decoder layers, %d samples",
                n_encoder, n_decoder, n_samples)

    src_path = get_src_path(args.data_dir, args.split, args.run)
    tgt_path = get_tgt_path(args.data_dir, args.split, args.run)
    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            sys.exit(EXIT_ERROR)

    labels = build_property_labels(src_path, tgt_path, args.data_dir)
    stem_final_match = labels["stem_final_match"]
    l_shaped = labels["l_shaped"]
    for name, y in (("stem_final_match", stem_final_match), ("l_shaped", l_shaped)):
        if len(y) != n_samples:
            logger.error("Sample count mismatch for %s: data=%d reps=%d",
                         name, len(y), n_samples)
            sys.exit(EXIT_ERROR)

    groups = build_lemma_groups(args.data_dir, args.split, args.run)

    # Log the subset composition up front (this is where the 10L_90NL caveat shows).
    for subset_name, sf_val in SUBSETS:
        m = np.ones(n_samples, dtype=bool) if sf_val is None else (stem_final_match == sf_val)
        n, n_L, n_NL, g_L, g_NL, base = subset_stats(l_shaped[m], groups[m])
        logger.info("Subset %-16s: n=%d L=%d (%d lemmas) NL=%d (%d lemmas) base=%.4f",
                    subset_name, n, n_L, g_L, n_NL, g_NL, base)

    layers = [("encoder", i) for i in range(n_encoder)] + [("decoder", i) for i in range(n_decoder)]
    rng = np.random.RandomState(config["probe"]["random_seed"])
    all_results = []
    for layer_type, layer_index in layers:
        rep_path = os.path.join(reps_dir, f"{layer_type}_layer_{layer_index}.pt")
        if not os.path.exists(rep_path):
            logger.warning("Skipping %s_%d: missing rep file", layer_type, layer_index)
            continue
        reps = torch.load(rep_path, weights_only=False)
        mask = load_pool_mask(reps_dir, layer_type, layer_index, args.pool_positions, logger=logger)
        if mask is None:
            logger.warning("Skipping %s_%d: missing mask file", layer_type, layer_index)
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
    logger.info("Saved %d results to %s", len(all_results), output_path)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
