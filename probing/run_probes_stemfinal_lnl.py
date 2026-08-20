"""Stem-final-match, conjugation-class and L/NL probing on all layers.

Probes whether each layer's representations encode three morphological
properties: stem_final_match (do the forms of one inflection instance share
the same stem-final consonant cluster; per sample), conjugation (-ar/-er/-ir;
lemma-level), and l_shaped (L vs NL morphome membership; lemma-level). The
metric is balanced accuracy because all three properties are heavily
class-imbalanced. Cross-validation folds default to lemma-disjoint so the
probe cannot memorize a lemma-to-class lookup.

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
                             level permutation of the lemma-to-label assignment;
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
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probing import EXIT_SKIPPED, MODEL_TYPES, SPLITS
from probing.extract_labels import (
    CONJUGATION_MAP,
    LSHAPED_MAP,
    build_conjugation_lookup,
    build_lshaped_lookup,
    stemfinal_match_label,
)
from probing.utils.cli import parse, standard_sentinels
from probing.utils.content_mask import load_pool_mask, pool_reps

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
    """Filename suffix encoding non-default protocol flags ('' for defaults)."""
    parts = []
    for key, default in VARIANT_DEFAULTS.items():
        val = getattr(args, key)
        if val != default:
            parts.append(key.replace("_", "") if val is True else f"{key.replace('_', '')}-{val}")
    return ("." + ".".join(parts)) if parts else ""


def shuffle_labels_by_group(y, groups, rng):
    """Control labels: permute which group (lemma) carries which label
    (a per-sample shuffle is trivially unlearnable under lemma-disjoint folds)."""
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
    with open(config_path) as f:
        return json.load(f)


def build_probe(probe_type, config):
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
    run_num = run.split("_")[0]
    return os.path.join(data_dir, split, "test", f"run{run_num}", f"test.{split}_{run}.tgt")


def get_src_path(data_dir, split, run):
    run_num = run.split("_")[0]
    return os.path.join(data_dir, split, "test", f"run{run_num}", f"test.{split}_{run}.src")


def build_property_labels(src_path, tgt_path, data_dir):
    """Build the per-sample stem_final_match, conjugation and l_shaped label arrays."""
    conj_lookup, conj_counts = build_conjugation_lookup(data_dir)
    lshaped_lookup, n_l_lemmas, n_nl_lemmas = build_lshaped_lookup(data_dir)

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
        print(f"WARNING: Lemma lookup misses (used fallback): {n_conj_miss} conjugation, {n_lnl_miss} l_shaped")

    return {
        "stem_final_match": np.array(stemfinal_labels, dtype=np.int64),
        "conjugation": np.array(conj_labels, dtype=np.int64),
        "l_shaped": np.array(lnl_labels, dtype=np.int64),
    }


def build_lemma_groups(data_dir, split, run):
    """Per-sample lemma id, so CV folds can be made lemma-disjoint."""
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
    """Probe one property on a single subset; return list of result row dicts."""
    cv_folds = config["probe"]["cv_folds"]
    seed = config["probe"]["random_seed"]
    scoring = ("balanced_accuracy", "accuracy")
    results = []

    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"WARNING:   {layer_type}_{layer_index} | {subset_name} | {property_name}: skipping (< 2 classes in {len(y)} samples)")
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
        print(f"WARNING:   {layer_type}_{layer_index} | {subset_name} | {property_name}: skipping (n_folds<2: classes={n_classes}, lemma groups={n_groups})")
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

    return results


if __name__ == "__main__":
    args = parse_args()

    # Check idempotency. Non-default protocol flags land in the filename so a
    # rerun with different settings can never be silently skipped against a
    # result file produced under other flags.
    output_dir = os.path.join(args.output_dir, args.model_type)
    variant = variant_tag(args)
    output_path = os.path.join(output_dir, f"{args.split}_{args.run}_stemfinal_lnl_results{variant}.csv")
    if os.path.exists(output_path):
        print(f"Skipping {args.model_type}/{args.split}_{args.run} -- results already exist")
        raise SystemExit(EXIT_SKIPPED)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    config = load_config(args.config)

    # Validate representations directory (or the pooled cache standing in for it)
    if args.pooled_cache_dir:
        reps_dir = os.path.join(args.pooled_cache_dir, args.model_type, f"{args.split}_{args.run}")
    else:
        reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Representations not found: {reps_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_encoder = metadata["n_encoder_layers"]
    n_decoder = metadata["n_decoder_layers"]
    n_samples = metadata["n_samples"]

    tgt_path = get_tgt_path(args.data_dir, args.split, args.run)
    src_path = get_src_path(args.data_dir, args.split, args.run)
    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    labels = build_property_labels(src_path, tgt_path, args.data_dir)

    for prop, y in labels.items():
        if len(y) != n_samples:
            raise ValueError(f"Sample count mismatch for {prop}: data={len(y)}, representations={n_samples}")

    for prop, y in labels.items():
        unique, counts = np.unique(y, return_counts=True)

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
        raise ValueError(f"Invalid --probe-types: {sorted(bad)}")

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
                raise FileNotFoundError(f"Pooled cache incomplete: missing {pooled_path}")
            X_all = np.load(pooled_path)
        else:
            rep_path = os.path.join(reps_dir, f"{layer_type}_layer_{layer_index}.pt")
            if not os.path.exists(rep_path):
                print(f"WARNING: Skipping {layer_type}_{layer_index}: missing rep file")
                continue
            reps = torch.load(rep_path, weights_only=False)

            mask = load_pool_mask(
                reps_dir,
                layer_type,
                layer_index,
                args.pool_positions,
            )
            if mask is None:
                print(f"WARNING: Skipping {layer_type}_{layer_index}: missing mask file")
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
