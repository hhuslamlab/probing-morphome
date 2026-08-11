"""Stem-final-match, conjugation-class and L/NL probing on all layers.

Probes whether layer representations encode three morphological properties:

  - stem_final_match: a *per-sample* binary flag -- do all the surface forms in
    one inflection instance (the source form(s) in the .src line plus the target
    form, i.e. the "three forms" of the dual-source setup) share the same
    stem-final consonant cluster?  'differ' marks an alternation across the
    forms (the surface footprint of the L-shape); 'same' marks none.
  - conjugation:      the three Spanish conjugation classes (-ar / -er / -ir),
    derived from each lemma's infinitive ending (lemma-level).
  - l_shaped:         whether the form's lemma belongs to an L-shaped paradigm
    (L vs NL), the L-shaped morphome membership (lemma-level).

The motivating question is whether, controlling for conjugation class and for
L/NL membership, the network represents whether the specific forms it was given
actually alternate at the stem-final consonant.  stem_final_match is computed
directly from the .src/.tgt surface strings, so it varies per training instance
rather than per lemma.

The readout is content mean-pooling over
the character positions of each layer (--pool-positions content, the leak-free
default), and cross-validation defaults to lemma-disjoint folds (--cv-mode
grouped) so the probe cannot memorise a lemma->class lookup across folds.

Metric is **balanced accuracy** (macro recall), NOT raw accuracy: all three
properties are heavily imbalanced (l_shaped ~89% NL, stem_final_match ~89%
'same', conjugation ~82% -ar), so a majority-class predictor scores ~0.82-0.89
on raw accuracy while balanced accuracy stays at chance (1/n_classes). Raw
accuracy is reported alongside for context, and --control adds a structure-
preserving selectivity control (see CONTROL_MODE); `selectivity` is computed on
balanced accuracy, `selectivity_raw` on raw accuracy.

Reuses the label lookups and helpers from extract_labels.py.

Usage:
  run_probes_stemfinal_lnl.py --model-type TYPE --split SPLIT --run RUN
                              [--representations-dir DIR] [--data-dir DIR]
                              [--output-dir DIR] [--config FILE]
                              [--control] [--n-controls N]
                              [--pool-positions POS] [--cv-mode MODE]
                              [--pooled-cache-dir DIR] [--probe-types LIST]
                              [--n-jobs N]
  run_probes_stemfinal_lnl.py (-h | --help)

Options:
  --model-type TYPE          Architecture (one of the five MODEL_TYPES).
  --split SPLIT              Data split, e.g. 10L_90NL.
  --run RUN                  Run identifier, e.g. 1_1.
  --representations-dir DIR  Directory with extracted representations
                             [default: data/probing/representations].
  --data-dir DIR             Root data directory containing split/test/runN/
                             folders; the raw test data lives in the
                             feature_informed repo [default: FEATURE_INFORMED_DATA].
  --output-dir DIR           Output directory for probe results
                             [default: data/probing/results_stemfinal_lnl_grouped].
  --config FILE              Probe config file [default: probing/config.json].
  --control                  Also run control probes with shuffled labels (for
                             selectivity). Lemma-level properties use a group-
                             level permutation of the lemma->label assignment;
                             stem_final_match uses a per-sample shuffle.
  --n-controls N             Label permutations to average the control over
                             [default: 5].
  --pool-positions POS       'content' (mean over chars, no tag leakage),
                             'last' (last content token), or 'all' (every
                             valid position; leaky, A/B only) [default: content].
  --cv-mode MODE             'grouped': lemma-disjoint StratifiedGroupKFold,
                             the leak-free protocol. 'per-form': ordinary
                             StratifiedKFold ignoring lemma — the leaky
                             baseline kept for the A/B contrast that
                             demonstrates lemma memorization [default: grouped].
  --pooled-cache-dir DIR     If set, load pre-pooled [n_samples, embed_dim]
                             .npy matrices from this cache (built by
                             pool_representations.py) instead of the full .pt
                             tensors. Scores are identical; only I/O differs.
  --probe-types LIST         Space-separated probe families to run ('linear',
                             'mlp'). Default linear only: the small MLP never
                             exceeded the linear probe on the pooled readout
                             and multiplies the sweep cost several-fold
                             [default: linear].
  --n-jobs N                 Parallel workers for cross-validation (folds run
                             concurrently; scores are identical to 1)
                             [default: 1].
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probing import EXIT_ERROR, EXIT_SKIPPED, EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.extract_labels import (
    CONJUGATION_MAP,
    LSHAPED_MAP,
    build_conjugation_lookup,
    build_lshaped_lookup,
    stemfinal_match_label,
)
from probing.utils.cli import parse, standard_sentinels
from probing.utils.content_mask import load_pool_mask, pool_reps

logger = logging.getLogger(__name__)

# Properties probed by this script, in output order.
PROPERTIES = ("stem_final_match", "conjugation", "l_shaped")

# How to build the shuffled-label control per property. Lemma-level properties
# get a group-level permutation (see shuffle_labels_by_group): under lemma-
# disjoint folds a per-sample shuffle is trivially unlearnable (control ~ chance
# by construction), so only the group-level permutation is an informative
# control there. stem_final_match varies within lemma, so it keeps the
# per-sample shuffle.
CONTROL_MODE = {"stem_final_match": "sample", "conjugation": "group", "l_shaped": "group"}

# Argument defaults that identify the canonical protocol; any non-default value
# is appended to the output filename (see variant_tag).
VARIANT_DEFAULTS = {
    "pool_positions": "content",
    "cv_mode": "grouped",
    "control": False,
    "probe_types": "linear",
}


def variant_tag(args):
    """Filename suffix encoding non-default protocol flags ('' for defaults).

    Keeps the all-defaults filename unchanged (backward compatible) while
    giving every flag combination its own output file, so the idempotency
    check can never silently skip a run with different settings.
    """
    parts = []
    for key, default in VARIANT_DEFAULTS.items():
        val = getattr(args, key)
        if val != default:
            parts.append(key.replace("_", "") if val is True else f"{key.replace('_', '')}-{val}")
    return ("." + ".".join(parts)) if parts else ""


def shuffle_labels_by_group(y, groups, rng):
    """Structure-preserving control labels: permute the per-GROUP assignment.

    Each group (lemma) keeps one consistent label and the multiset of
    per-group labels is preserved; only which group carries which label is
    randomized. This is the control a lemma-disjoint probe could actually
    exploit (cf. Hewitt & Liang 2019), unlike a per-sample shuffle, which
    destroys the group<->label structure entirely. Only meaningful for
    group-constant properties.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    group_labels = np.empty(len(uniq), dtype=y.dtype)
    group_labels[inv] = y  # one write per sample; group-constant by assumption
    perm = rng.permutation(len(uniq))
    return group_labels[perm][inv]


def parse_args():
    return parse(__doc__,
                 types=dict(n_controls=int, n_jobs=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS,
                              pool_positions=("content", "all", "last"),
                              cv_mode=("grouped", "per-form")),
                 sentinels=standard_sentinels())


def load_config(config_path):
    """Load probing configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def build_probe(probe_type, config):
    """Build an sklearn pipeline for the given probe type."""
    probe_config = config["probe"]

    if probe_type == "linear":
        clf = LogisticRegression(
            C=probe_config["linear"]["C"],
            solver=probe_config["linear"]["solver"],
            max_iter=probe_config["linear"]["max_iter"],
            random_state=probe_config["random_seed"],
        )
    elif probe_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(probe_config["mlp"]["hidden_layer_sizes"]),
            max_iter=probe_config["mlp"]["max_iter"],
            random_state=probe_config["random_seed"],
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_tgt_path(data_dir, split, run):
    """Construct the test .tgt file path."""
    run_num = run.split("_")[0]
    return os.path.join(data_dir, split, "test", f"run{run_num}", f"test.{split}_{run}.tgt")


def get_src_path(data_dir, split, run):
    """Construct the test .src file path."""
    run_num = run.split("_")[0]
    return os.path.join(data_dir, split, "test", f"run{run_num}", f"test.{split}_{run}.src")


def build_property_labels(src_path, tgt_path, data_dir):
    """Build per-sample stem_final_match, conjugation and l_shaped labels.

    conjugation and l_shaped are lemma-level, keyed on (target_tag,
    normalized_form) via the same lookups extract_labels.py uses (same
    fallbacks: -ar for unknown conjugation, NL for unknown L-shape).
    stem_final_match is per-sample, computed directly from the src/tgt surface
    forms by stemfinal_match_label.

    Returns:
        labels: dict mapping property name -> [n_samples] int array.
    """
    conj_lookup, conj_counts = build_conjugation_lookup(data_dir)
    lshaped_lookup, n_l_lemmas, n_nl_lemmas = build_lshaped_lookup(data_dir)
    logger.info("Conjugation classes (lemmas): %s", dict(conj_counts))
    logger.info("L-shaped lookup: %d L lemmas, %d NL lemmas", n_l_lemmas, n_nl_lemmas)

    with open(src_path) as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path) as f:
        tgt_lines = [line.strip() for line in f]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {len(src_lines)} src vs {len(tgt_lines)} tgt")

    stemfinal_labels = []
    conj_labels = []
    lnl_labels = []
    n_conj_miss = 0
    n_lnl_miss = 0
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        # Per-sample: do src form(s) + target form share one stem-final cluster?
        stemfinal_labels.append(stemfinal_match_label(src_line, tgt_line))

        # Source format: form1 <tag1> # form2 <tag2> # <target_tag>
        target_tag_bare = src_line.split(" # ")[-1].strip().strip("<>")
        normalized_form = tgt_line.replace(" ", "")
        key = (target_tag_bare, normalized_form)

        if key in conj_lookup:
            conj = conj_lookup[key]
        else:
            conj = "ar"  # fallback to -ar (dominant class), mirrors extract_labels
            n_conj_miss += 1
        conj_labels.append(CONJUGATION_MAP[conj])

        if key in lshaped_lookup:
            is_l = lshaped_lookup[key]
        else:
            is_l = False  # fallback to NL, mirrors extract_labels
            n_lnl_miss += 1
        lnl_labels.append(LSHAPED_MAP["L"] if is_l else LSHAPED_MAP["NL"])

    if n_conj_miss or n_lnl_miss:
        logger.warning(
            "Lemma lookup misses (used fallback): %d conjugation, %d l_shaped",
            n_conj_miss,
            n_lnl_miss,
        )

    return {
        "stem_final_match": np.array(stemfinal_labels, dtype=np.int64),
        "conjugation": np.array(conj_labels, dtype=np.int64),
        "l_shaped": np.array(lnl_labels, dtype=np.int64),
    }


def build_lemma_groups(data_dir, split, run):
    """Per-sample lemma id, so CV folds can be made lemma-disjoint.

    Without this, the same lemma's cells land in both train and test folds and
    the probe memorises the (deterministic) lemma->class lookup instead of
    testing whether the property generalises to unseen lemmas.  Keyed on the
    TARGET form's lemma via lemma_form.json (a list of {lemma: {tag: form}}).
    Syncretic forms can merge two lemmas into one group — safe (never splits a
    lemma across folds, only conservative).  Returns a fresh singleton group
    where the form is not found in the lemma table.
    """
    run_num = run.split("_")[0]
    base = os.path.join(data_dir, split, "test", f"run{run_num}")
    with open(os.path.join(base, "lemma_form.json")) as f:
        lemma_form = json.load(f)  # list of {lemma: {tag: form}}
    form2lemma = {}
    for li, entry in enumerate(lemma_form):
        for lemma, forms in entry.items():
            form2lemma[lemma.replace(" ", "")] = li
            for form in forms.values():
                form2lemma.setdefault(form.replace(" ", ""), li)
    groups = []
    next_singleton = len(lemma_form)
    with open(get_tgt_path(data_dir, split, run)) as f:
        for line in f:
            form = line.strip().replace(" ", "")
            if form in form2lemma:
                groups.append(form2lemma[form])
            else:
                groups.append(next_singleton)
                next_singleton += 1
    return np.array(groups, dtype=np.int64)


def run_probe_on_subset(
    X,
    y,
    groups,
    subset_name,
    layer_type,
    layer_index,
    config,
    control,
    rng,
    probe_site,
    property_name,
    cv_mode="grouped",
    n_jobs=1,
    n_controls=5,
    probe_types=("linear",),
):
    """Run probes on a single subset, return list of result row dicts.

    cv_mode='grouped' (default): lemma-disjoint folds (StratifiedGroupKFold) so the
    probe cannot memorise the lemma->class lookup across folds. cv_mode='per-form':
    ordinary StratifiedKFold (rows split at random, ignoring lemma) — the leaky
    baseline, kept for the A/B contrast that demonstrates the leakage.

    The control (when requested) averages n_controls label permutations; the
    permutation is group-level or per-sample per CONTROL_MODE[property_name].

    The headline metric is **balanced accuracy** (macro recall), not raw
    accuracy, and selectivity is computed on it. Every property here is heavily
    imbalanced (l_shaped ~89% NL, stem_final_match ~89% 'same'), so raw accuracy
    mostly measures the class prior: a majority-class predictor scores ~0.89
    while balanced accuracy stays at chance (1/n_classes). Raw accuracy is still
    reported alongside for context. Same rationale as
    run_probes_lnl_within_stemfinal.py.
    """
    cv_folds = config["probe"]["cv_folds"]
    seed = config["probe"]["random_seed"]
    scoring = ("balanced_accuracy", "accuracy")
    results = []

    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.warning(
            "  %s_%d | %s | %s: skipping (< 2 classes in %d samples)",
            layer_type,
            layer_index,
            subset_name,
            property_name,
            len(y),
        )
        return results

    # Cap folds by the sparsest class: StratifiedKFold needs n_splits <= the
    # smallest class's sample count, StratifiedGroupKFold needs each class
    # spread over at least n_splits lemma groups to stratify.
    classes, class_counts = np.unique(y, return_counts=True)
    if cv_mode == "per-form":
        n_folds = min(cv_folds, int(class_counts.min()))
    else:
        per_class_groups = min(len(np.unique(groups[y == c])) for c in classes)
        n_folds = min(cv_folds, per_class_groups)
    n_groups = len(np.unique(groups))
    if n_folds < 2:
        logger.warning(
            "  %s_%d | %s | %s: skipping (n_folds<2: classes=%d, lemma groups=%d)",
            layer_type,
            layer_index,
            subset_name,
            property_name,
            n_classes,
            n_groups,
        )
        return results

    _, counts = np.unique(y, return_counts=True)
    majority_baseline = counts.max() / counts.sum()

    for probe_type in probe_types:
        pipe = build_probe(probe_type, config)
        if cv_mode == "per-form":
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            real = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        else:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            real = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs)
        bal_scores = real["test_balanced_accuracy"]
        raw_scores = real["test_accuracy"]
        bal, bal_std = bal_scores.mean(), bal_scores.std()
        acc, std = raw_scores.mean(), raw_scores.std()

        row = {
            "layer_type": layer_type,
            "layer_index": layer_index,
            "subset": subset_name,
            "probe_site": probe_site,
            "property": property_name,
            "probe_type": probe_type,
            "balanced_accuracy": bal,
            "balanced_std": bal_std,
            "accuracy": acc,
            "std": std,
            "chance": 1.0 / n_classes,
            "majority_baseline": majority_baseline,
            "n_classes": n_classes,
            "n_samples": len(y),
        }

        if control:
            control_mode = CONTROL_MODE.get(property_name, "sample")
            ctrl_bal_means, ctrl_raw_means = [], []
            for _ in range(n_controls):
                if control_mode == "group":
                    y_shuffled = shuffle_labels_by_group(y, groups, rng)
                else:
                    y_shuffled = y.copy()
                    rng.shuffle(y_shuffled)
                if len(np.unique(y_shuffled)) < 2:
                    continue  # degenerate permutation (all groups same label)
                pipe_ctrl = build_probe(probe_type, config)
                if cv_mode == "per-form":
                    cv_ctrl = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    ctrl = cross_validate(pipe_ctrl, X, y_shuffled, cv=cv_ctrl, scoring=scoring, n_jobs=n_jobs)
                else:
                    cv_ctrl = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    ctrl = cross_validate(pipe_ctrl, X, y_shuffled, groups=groups, cv=cv_ctrl, scoring=scoring, n_jobs=n_jobs)
                ctrl_bal_means.append(ctrl["test_balanced_accuracy"].mean())
                ctrl_raw_means.append(ctrl["test_accuracy"].mean())
            # selectivity is on BALANCED accuracy — the headline metric.
            row["control_balanced_accuracy"] = float(np.mean(ctrl_bal_means))
            row["control_balanced_std"] = float(np.std(ctrl_bal_means))
            row["selectivity"] = bal - row["control_balanced_accuracy"]
            row["control_accuracy"] = float(np.mean(ctrl_raw_means))
            row["control_std"] = float(np.std(ctrl_raw_means))
            row["selectivity_raw"] = acc - row["control_accuracy"]
            row["control_mode"] = control_mode
            row["n_controls"] = len(ctrl_bal_means)

        results.append(row)

        log_msg = (
            f"  {layer_type}_{layer_index} | {subset_name} | {property_name} | {probe_type}: "
            f"bal_acc={bal:.4f} (+/- {bal_std:.4f}) [chance={1.0/n_classes:.4f}, "
            f"raw={acc:.4f}, base={majority_baseline:.4f}, n={len(y)}]"
        )
        if control:
            log_msg += (
                f" [ctrl_bal={row['control_balanced_accuracy']:.4f} "
                f"sel={row['selectivity']:+.4f} ({row['control_mode']})]"
            )
        logger.info(log_msg)

    return results


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    # Check idempotency. Non-default protocol flags land in the filename so a
    # rerun with different settings can never be silently skipped against a
    # result file produced under other flags.
    output_dir = os.path.join(args.output_dir, args.model_type)
    variant = variant_tag(args)
    output_path = os.path.join(output_dir, f"{args.split}_{args.run}_stemfinal_lnl_results{variant}.csv")
    logger.info("Output file: %s", output_path)
    if os.path.exists(output_path):
        logger.info(
            "Skipping %s/%s_%s -- results already exist",
            args.model_type,
            args.split,
            args.run,
        )
        sys.exit(EXIT_SKIPPED)

    if not os.path.exists(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(EXIT_ERROR)
    config = load_config(args.config)

    # Validate representations directory (or the pooled cache standing in for it)
    if args.pooled_cache_dir:
        reps_dir = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")
    else:
        reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error("Representations not found: %s", reps_dir)
        sys.exit(EXIT_ERROR)

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_encoder = metadata["n_encoder_layers"]
    n_decoder = metadata["n_decoder_layers"]
    n_samples = metadata["n_samples"]
    logger.info(
        "Loaded metadata: %d encoder + %d decoder layers, %d samples",
        n_encoder,
        n_decoder,
        n_samples,
    )

    tgt_path = get_tgt_path(args.data_dir, args.split, args.run)
    src_path = get_src_path(args.data_dir, args.split, args.run)
    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            sys.exit(EXIT_ERROR)

    labels = build_property_labels(src_path, tgt_path, args.data_dir)

    for prop, y in labels.items():
        if len(y) != n_samples:
            logger.error(
                "Sample count mismatch for %s: data=%d, representations=%d",
                prop,
                len(y),
                n_samples,
            )
            sys.exit(EXIT_ERROR)

    for prop, y in labels.items():
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Labels[%s]: %s", prop, dict(zip(unique.tolist(), counts.tolist())))

    # Per-sample lemma id for lemma-disjoint CV folds.
    lemma_groups = build_lemma_groups(args.data_dir, args.split, args.run)

    layers = []
    for i in range(n_encoder):
        layers.append(("encoder", i))
    for i in range(n_decoder):
        layers.append(("decoder", i))

    probe_types = args.probe_types.split()
    bad = set(probe_types) - {"linear", "mlp"}
    if bad:
        logger.error("Invalid --probe-types: %s", sorted(bad))
        sys.exit(EXIT_ERROR)

    seed = config["probe"]["random_seed"]
    rng = np.random.RandomState(seed)

    all_results = []

    for layer_type, layer_index in layers:
        if args.pooled_cache_dir:
            # Stage-1 pooling (pool_representations.py) guarantees a complete
            # cache, so a missing .npy means a broken cache: hard error rather
            # than warn-and-continue, which would silently lock in an
            # incomplete CSV via the idempotency skip.
            pooled_path = os.path.join(
                reps_dir, f"{layer_type}_layer_{layer_index}_{args.pool_positions}.npy"
            )
            if not os.path.exists(pooled_path):
                logger.error("Pooled cache incomplete: missing %s", pooled_path)
                sys.exit(EXIT_ERROR)
            X_all = np.load(pooled_path)
        else:
            rep_path = os.path.join(reps_dir, f"{layer_type}_layer_{layer_index}.pt")
            if not os.path.exists(rep_path):
                logger.warning("Skipping %s_%d: missing rep file", layer_type, layer_index)
                continue
            reps = torch.load(rep_path, weights_only=False)

            mask = load_pool_mask(
                reps_dir,
                layer_type,
                layer_index,
                args.pool_positions,
                logger=logger,
            )
            if mask is None:
                logger.warning("Skipping %s_%d: missing mask file", layer_type, layer_index)
                continue
            X_all = pool_reps(reps, mask, args.pool_positions).numpy()
        probe_site = "content-pool" if args.pool_positions == "content" else f"pool-{args.pool_positions}"

        for prop in PROPERTIES:
            rows = run_probe_on_subset(
                X_all,
                labels[prop],
                lemma_groups,
                "all",
                layer_type,
                layer_index,
                config,
                args.control,
                rng,
                probe_site,
                prop,
                cv_mode=args.cv_mode,
                n_jobs=args.n_jobs,
                n_controls=args.n_controls,
                probe_types=probe_types,
            )
            all_results.extend(rows)

    # Write CSV. balanced_accuracy leads (the headline metric); raw accuracy and
    # the majority baseline follow for context. Control columns only exist when
    # --control ran.
    os.makedirs(output_dir, exist_ok=True)

    columns = [
        "layer_type", "layer_index", "subset", "probe_site", "property", "probe_type",
        "balanced_accuracy", "balanced_std", "accuracy", "std",
    ]
    if args.control:
        columns += [
            "control_balanced_accuracy", "control_balanced_std", "selectivity",
            "control_accuracy", "control_std", "selectivity_raw",
            "control_mode", "n_controls",
        ]
    columns += ["chance", "majority_baseline", "n_classes", "n_samples"]

    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    with open(output_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in all_results:
            f.write(",".join(fmt(row[c]) for c in columns) + "\n")

    logger.info("Saved %d results to %s", len(all_results), output_path)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
