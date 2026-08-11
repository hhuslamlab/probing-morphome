"""Surface n-gram baselines for the stem-final / conjugation / L-NL probes.

How much of what the probes decode is predictable from SURFACE FORM alone?
This script re-runs the probing protocols of run_probes_stemfinal_lnl.py
(analysis A) and run_probes_lnl_within_stemfinal.py (analysis B) with the
model representations replaced by surface n-gram statistics of the very same
inputs -- same labels, same lemma-disjoint StratifiedGroupKFold folds (same
seed, same y/groups => identical splits), same metrics -- so every row is
directly comparable to a probe row.

Two baseline families (--baselines):

  - classifier: phoneme n-gram counts (CountVectorizer over the space-
    separated phoneme tokens, orders 1..n) fed to the same LogisticRegression
    the probes use. Two deliberate deviations from build_probe: no
    StandardScaler (mean-centering is meaningless on sparse counts), and the
    vectorizer sits INSIDE the CV pipeline so the n-gram vocabulary is refit
    per fold (no vocabulary leakage across folds).
  - lm: class-conditional generative phoneme n-gram LMs; predict = argmax
    class log-likelihood under a UNIFORM class prior, i.e. a pure likelihood
    ratio that cannot fall back on majority-class guessing (the
    majority_baseline column is still reported for reference). Skips
    stem_final_match: that label is a relation BETWEEN the forms of one
    instance, which a per-class generative model of strings does not model
    coherently (the classifier family covers it).

Text views (probe_site column) come from utils/surface_text.py: src-content /
tgt-content mirror what content pooling exposes to encoder / decoder probes;
all-content adds the src+tgt union (decoder states attend to the encoder, so
this is the fair analog of decoder probes and the only view with full
information for stem_final_match); src-with-tags (--with-tags) mirrors tag
pooling.

Representation-free: runs per (split, run) only, no --model-type. Results land
under the pseudo-arch directory ``ngram/`` in the same two CSV schemas as the
probe scripts (layer_type='ngram', layer_index=n-gram order), so the
summarizers can overlay them.

Usage:
  run_ngram_baselines.py --split SPLIT --run RUN [--data-dir DIR]
                         [--output-dir DIR] [--config FILE]
                         [--ngram-orders ORDERS] [--baselines FAMILIES]
                         [--with-tags] [--control] [--n-controls N]
                         [--cv-mode MODE] [--n-jobs N]
  run_ngram_baselines.py (-h | --help)

Options:
  --split SPLIT         Data split, e.g. 10L_90NL.
  --run RUN             Run identifier, e.g. 1_2.
  --data-dir DIR        Root data directory containing split/test/runN/
                        folders [default: FEATURE_INFORMED_DATA].
  --output-dir DIR      Output root; CSVs land under <output-dir>/ngram/
                        [default: data/probing/results_ngram_baselines_balanced].
  --config FILE         Probe config file [default: probing/config.json].
  --ngram-orders ORDERS  Space-separated max n-gram orders to evaluate; each
                         order n uses n-grams 1..n [default: 1 2 3].
  --baselines FAMILIES  Space-separated baseline families to run
                        [default: classifier lm].
  --with-tags           Also evaluate the src-with-tags view (leaks the
                        morphological tags; analog of tag pooling).
  --control             Analysis A only: also run shuffled-label controls
                        (analysis B always includes its group-level control,
                        matching run_probes_lnl_within_stemfinal).
  --n-controls N        Label permutations for the control [default: 5].
  --cv-mode MODE        Analysis A folds: 'grouped' for lemma-disjoint or the
                        leaky 'per-form' baseline [default: grouped].
  --n-jobs N            Parallel workers for cross-validation [default: 1].
"""

import logging
import math
import os
import sys
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from probing import EXIT_ERROR, EXIT_SKIPPED, EXIT_SUCCESS, SPLITS
from probing.utils.cli import parse, standard_sentinels
from probing.run_probes_lnl_within_stemfinal import (
    CSV_COLUMNS as WITHIN_CSV_COLUMNS,
    SUBSETS,
    subset_stats,
)
from probing.run_probes_stemfinal_lnl import (
    CONTROL_MODE,
    PROPERTIES,
    build_lemma_groups,
    build_property_labels,
    get_src_path,
    get_tgt_path,
    load_config,
    shuffle_labels_by_group,
)
from probing.utils.surface_text import build_texts

logger = logging.getLogger(__name__)

# Short view names used inside probe_type labels of the within-stemfinal CSV
# (that schema has no probe_site column).
VIEW_SHORT = {
    "src-content": "src",
    "tgt-content": "tgt",
    "all-content": "all",
    "src-with-tags": "srctags",
}

BASELINE_FAMILIES = ("classifier", "lm")

# probe_type values for analysis A rows, per family.
FAMILY_PROBE_TYPE = {"classifier": "ngram-linear", "lm": "ngram-lm"}

# Argument defaults that identify the canonical protocol; any non-default value
# is appended to the output filenames (mirrors variant_tag in
# run_probes_stemfinal_lnl.py so an idempotency skip can never hide a rerun
# with different settings).
VARIANT_DEFAULTS = {
    "cv_mode": "grouped",
    "control": False,
    "with_tags": False,
    "ngram_orders": (1, 2, 3),
    "baselines": ("classifier", "lm"),
}


class NgramLMClassifier(BaseEstimator, ClassifierMixin):
    """Class-conditional phoneme n-gram LM classifier (pure likelihood ratio).

    fit() counts n-grams of orders 1..order per class (BOS padding, EOS
    terminator). predict() scores each text under every class LM with stupid
    backoff -- the highest order whose n-gram was seen wins, discounted by
    backoff_alpha per backed-off level, with an add-k floor at the unigram
    level for unseen symbols -- and returns the argmax class under a uniform
    class prior.
    """

    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, order=3, add_k=0.1, backoff_alpha=0.4):
        self.order = order
        self.add_k = add_k
        self.backoff_alpha = backoff_alpha

    def _pad(self, text):
        return [self.BOS] * (self.order - 1) + text.split() + [self.EOS]

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.vocab_ = set()
        self.counts_ = {}
        for c in self.classes_:
            ngrams, contexts = Counter(), Counter()
            for text in X[y == c]:
                toks = self._pad(text)
                self.vocab_.update(toks[self.order - 1 :])
                for i in range(self.order - 1, len(toks)):
                    for n in range(1, self.order + 1):
                        ngram = tuple(toks[i - n + 1 : i + 1])
                        ngrams[ngram] += 1
                        contexts[ngram[:-1]] += 1
            self.counts_[c] = (ngrams, contexts)
        return self

    def _logprob(self, toks, ngrams, contexts):
        lp = 0.0
        v = len(self.vocab_) + 1  # +1 mass slot for unseen symbols
        for i in range(self.order - 1, len(toks)):
            p = None
            for n in range(self.order, 0, -1):
                ngram = tuple(toks[i - n + 1 : i + 1])
                c = ngrams.get(ngram, 0)
                if c > 0:
                    p = (self.backoff_alpha ** (self.order - n)) * c / contexts[ngram[:-1]]
                    break
            if p is None:  # unseen even as a unigram
                p = (
                    (self.backoff_alpha ** (self.order - 1))
                    * self.add_k
                    / (contexts[()] + self.add_k * v)
                )
            lp += math.log(p)
        return lp

    def predict(self, X):
        X = np.asarray(X, dtype=object)
        preds = np.empty(len(X), dtype=self.classes_.dtype)
        for idx, text in enumerate(X):
            toks = self._pad(text)
            scores = [self._logprob(toks, *self.counts_[c]) for c in self.classes_]
            preds[idx] = self.classes_[int(np.argmax(scores))]
        return preds


def build_ngram_classifier(order, config):
    """Surface analog of build_probe: n-gram counts -> the probes' logistic.

    Word-level n-grams over the space-separated phoneme tokens (analyzer='char'
    would split multi-codepoint IPA symbols and glue phonemes across spaces).
    """
    lin = config["probe"]["linear"]
    return Pipeline(
        [
            (
                "vec",
                CountVectorizer(
                    analyzer="word",
                    tokenizer=str.split,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, order),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=lin["C"],
                    solver=lin["solver"],
                    max_iter=lin["max_iter"],
                    random_state=config["probe"]["random_seed"],
                ),
            ),
        ]
    )


def make_estimator(family, order, config):
    if family == "classifier":
        return build_ngram_classifier(order, config)
    return NgramLMClassifier(order=order)


def variant_tag(args):
    """Filename suffix encoding non-default protocol flags ('' for defaults)."""
    parts = []
    for key, default in VARIANT_DEFAULTS.items():
        val = getattr(args, key)
        if isinstance(default, tuple):
            val = tuple(val)
        if val != default:
            if val is True:
                parts.append(key.replace("_", ""))
            elif isinstance(val, tuple):
                parts.append(f"{key.replace('_', '')}-{'-'.join(str(v) for v in val)}")
            else:
                parts.append(f"{key.replace('_', '')}-{val}")
    return ("." + ".".join(parts)) if parts else ""


def parse_args():
    args = parse(__doc__,
                 types=dict(n_controls=int, n_jobs=int),
                 choices=dict(split=SPLITS, cv_mode=("grouped", "per-form")),
                 sentinels=standard_sentinels())
    args.ngram_orders = [int(n) for n in args.ngram_orders.split()]
    args.baselines = args.baselines.split()
    bad = set(args.baselines) - set(BASELINE_FAMILIES)
    if bad:
        raise SystemExit(f"invalid --baselines: {sorted(bad)} (choose from {list(BASELINE_FAMILIES)})")
    return args


def run_baseline_rows(
    X_texts,
    y,
    groups,
    order,
    probe_site,
    property_name,
    family,
    config,
    cv_mode,
    control,
    rng,
    n_controls,
    n_jobs,
):
    """Analysis A: one (view, order, family, property) cell.

    Mirrors run_probes_stemfinal_lnl.run_probe_on_subset (fold caps, majority
    baseline, control modes) with the pooled representations replaced by the
    surface texts.
    """
    cv_folds = config["probe"]["cv_folds"]
    seed = config["probe"]["random_seed"]
    probe_type = FAMILY_PROBE_TYPE[family]

    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.warning("  ngram_%d | %s | %s: skipping (< 2 classes)", order, probe_site, property_name)
        return []
    # Cap folds by the sparsest class (per-class sample count for plain
    # stratified folds, per-class lemma-group count for grouped folds) — NOT by
    # n_classes, which would silently cap binary properties at 2 folds.
    classes, class_counts = np.unique(y, return_counts=True)
    if cv_mode == "per-form":
        n_folds = min(cv_folds, int(class_counts.min()))
    else:
        per_class_groups = min(len(np.unique(groups[y == c])) for c in classes)
        n_folds = min(cv_folds, per_class_groups)
    if n_folds < 2:
        logger.warning("  ngram_%d | %s | %s: skipping (n_folds<2)", order, probe_site, property_name)
        return []

    _, counts = np.unique(y, return_counts=True)
    majority_baseline = counts.max() / counts.sum()

    est = make_estimator(family, order, config)
    # Balanced accuracy is the headline metric (matches the probes, which report
    # it for the same heavily imbalanced properties); raw accuracy is kept for
    # backward compatibility with earlier baseline CSVs.
    if cv_mode == "per-form":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        res = cross_validate(
            est, X_texts, y, cv=cv,
            scoring=("balanced_accuracy", "accuracy"), n_jobs=n_jobs,
        )
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        res = cross_validate(
            est, X_texts, y, groups=groups, cv=cv,
            scoring=("balanced_accuracy", "accuracy"), n_jobs=n_jobs,
        )
    bal = res["test_balanced_accuracy"]
    scores = res["test_accuracy"]
    acc, std = scores.mean(), scores.std()

    row = {
        "layer_type": "ngram",
        "layer_index": order,
        "subset": "all",
        "probe_site": probe_site,
        "property": property_name,
        "probe_type": probe_type,
        "balanced_accuracy": bal.mean(),
        "balanced_std": bal.std(),
        "accuracy": acc,
        "std": std,
        "majority_baseline": majority_baseline,
        "n_classes": n_classes,
        "n_samples": len(y),
    }

    if control:
        control_mode = CONTROL_MODE.get(property_name, "sample")
        ctrl_means = []
        for _ in range(n_controls):
            if control_mode == "group":
                y_shuffled = shuffle_labels_by_group(y, groups, rng)
            else:
                y_shuffled = y.copy()
                rng.shuffle(y_shuffled)
            if len(np.unique(y_shuffled)) < 2:
                continue
            est_ctrl = make_estimator(family, order, config)
            if cv_mode == "per-form":
                cv_ctrl = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                ctrl_scores = cross_val_score(
                    est_ctrl, X_texts, y_shuffled, cv=cv_ctrl, scoring="accuracy", n_jobs=n_jobs
                )
            else:
                cv_ctrl = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                ctrl_scores = cross_val_score(
                    est_ctrl,
                    X_texts,
                    y_shuffled,
                    groups=groups,
                    cv=cv_ctrl,
                    scoring="accuracy",
                    n_jobs=n_jobs,
                )
            ctrl_means.append(ctrl_scores.mean())
        row["control_accuracy"] = float(np.mean(ctrl_means))
        row["control_std"] = float(np.std(ctrl_means))
        row["selectivity"] = acc - row["control_accuracy"]
        row["control_mode"] = control_mode
        row["n_controls"] = len(ctrl_means)

    log_msg = (
        f"  ngram_{order} | {probe_site} | {property_name} | {probe_type}: "
        f"{acc:.4f} (+/- {std:.4f}) [baseline={majority_baseline:.4f}, n={len(y)}]"
    )
    if control:
        log_msg += f" [control={row['control_accuracy']:.4f}]"
    logger.info(log_msg)
    return [row]


def within_subset_rows(
    X_texts, y, groups, subset_name, order, probe_type_label, family, config, rng, n_controls, n_jobs
):
    """Analysis B: l_shaped on one stem_final_match subset.

    Mirrors run_probes_lnl_within_stemfinal.probe_subset: balanced accuracy via
    lemma-disjoint folds capped by the rarer class's lemma count, plus the
    group-level shuffled-label control (always on -- the CSV schema requires it).
    """
    n, n_L, n_NL, g_L, g_NL, majority = subset_stats(y, groups)
    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.warning("  ngram_%d | %s | %s: skipping (only %d class)", order, subset_name, probe_type_label, n_classes)
        return []
    n_folds = min(config["probe"]["cv_folds"], min(g_L, g_NL))
    if n_folds < 2:
        logger.warning(
            "  ngram_%d | %s | %s: skipping (n_folds<2: L lemmas=%d, NL lemmas=%d)",
            order, subset_name, probe_type_label, g_L, g_NL,
        )
        return []

    seed = config["probe"]["random_seed"]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    real = cross_validate(
        make_estimator(family, order, config),
        X_texts,
        y,
        groups=groups,
        cv=cv,
        scoring=("balanced_accuracy", "accuracy"),
        n_jobs=n_jobs,
    )
    bal = real["test_balanced_accuracy"]
    raw = real["test_accuracy"]

    ctrl_bals = []
    for _ in range(n_controls):
        y_shuf = shuffle_labels_by_group(y, groups, rng)
        if len(np.unique(y_shuf)) < 2:
            continue
        cv_ctrl = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        ctrl = cross_validate(
            make_estimator(family, order, config),
            X_texts,
            y_shuf,
            groups=groups,
            cv=cv_ctrl,
            scoring=("balanced_accuracy",),
            n_jobs=n_jobs,
        )
        ctrl_bals.append(ctrl["test_balanced_accuracy"].mean())
    ctrl_bal = float(np.mean(ctrl_bals))
    ctrl_bal_std = float(np.std(ctrl_bals))

    row = {
        "layer_type": "ngram",
        "layer_index": order,
        "subset": subset_name,
        "probe_type": probe_type_label,
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
    logger.info(
        "  ngram_%d | %-16s | %-12s: bal_acc=%.4f (+/-%.4f) ctrl=%.4f sel=%+.4f "
        "[raw=%.4f base=%.4f n=%d L=%d/%dlem folds=%d]",
        order, subset_name, probe_type_label,
        bal.mean(), bal.std(), ctrl_bal, bal.mean() - ctrl_bal,
        raw.mean(), majority, n, n_L, g_L, n_folds,
    )
    return [row]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    output_dir = os.path.join(args.output_dir, "ngram")
    variant = variant_tag(args)
    path_a = os.path.join(output_dir, f"{args.split}_{args.run}_stemfinal_lnl_results{variant}.csv")
    path_b = os.path.join(output_dir, f"{args.split}_{args.run}_lnl_within_stemfinal{variant}.csv")
    logger.info("Output files: %s , %s", path_a, path_b)
    if os.path.exists(path_a) and os.path.exists(path_b):
        logger.info("Skipping ngram/%s_%s -- results already exist", args.split, args.run)
        sys.exit(EXIT_SKIPPED)

    if not os.path.exists(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(EXIT_ERROR)
    config = load_config(args.config)

    src_path = get_src_path(args.data_dir, args.split, args.run)
    tgt_path = get_tgt_path(args.data_dir, args.split, args.run)
    for path in (src_path, tgt_path):
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            sys.exit(EXIT_ERROR)

    labels = build_property_labels(src_path, tgt_path, args.data_dir)
    for prop, y in labels.items():
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Labels[%s]: %s", prop, dict(zip(unique.tolist(), counts.tolist())))

    with open(src_path) as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path) as f:
        tgt_lines = [line.strip() for line in f]
    views = build_texts(src_lines, tgt_lines, with_tags=args.with_tags)
    for name, texts in views.items():
        if len(texts) != len(labels["l_shaped"]):
            logger.error("View %s has %d rows, labels have %d", name, len(texts), len(labels["l_shaped"]))
            sys.exit(EXIT_ERROR)

    groups = build_lemma_groups(args.data_dir, args.split, args.run)
    seed = config["probe"]["random_seed"]
    rng = np.random.RandomState(seed)

    # Analysis A: full-set accuracy per property (probe-row schema)
    logger.info("=== Analysis A: full-set n-gram baselines ===")
    rows_a = []
    for probe_site, texts in views.items():
        X_texts = np.array(texts, dtype=object)
        for order in args.ngram_orders:
            for family in args.baselines:
                for prop in PROPERTIES:
                    if family == "lm" and prop == "stem_final_match":
                        continue  # relational label; see module docstring
                    rows_a.extend(
                        run_baseline_rows(
                            X_texts,
                            labels[prop],
                            groups,
                            order,
                            probe_site,
                            prop,
                            family,
                            config,
                            args.cv_mode,
                            args.control,
                            rng,
                            args.n_controls,
                            args.n_jobs,
                        )
                    )

    # Analysis B: l_shaped within stem_final_match subsets
    logger.info("=== Analysis B: L/NL within stem-final subsets ===")
    stem_final_match = labels["stem_final_match"]
    l_shaped = labels["l_shaped"]
    rows_b = []
    for probe_site, texts in views.items():
        X_texts = np.array(texts, dtype=object)
        for order in args.ngram_orders:
            for family in args.baselines:
                prefix = "ngram" if family == "classifier" else "lm"
                probe_type_label = f"{prefix}{order}-{VIEW_SHORT[probe_site]}"
                for subset_name, sf_val in SUBSETS:
                    m = (
                        np.ones(len(l_shaped), dtype=bool)
                        if sf_val is None
                        else (stem_final_match == sf_val)
                    )
                    rows_b.extend(
                        within_subset_rows(
                            X_texts[m],
                            l_shaped[m],
                            groups[m],
                            subset_name,
                            order,
                            probe_type_label,
                            family,
                            config,
                            rng,
                            args.n_controls,
                            args.n_jobs,
                        )
                    )

    os.makedirs(output_dir, exist_ok=True)

    # Analysis A CSV: identical schema to run_probes_stemfinal_lnl.py.
    if args.control:
        header = (
            "layer_type,layer_index,subset,probe_site,property,probe_type,accuracy,std,"
            "control_accuracy,control_std,selectivity,control_mode,n_controls,"
            "majority_baseline,n_classes,n_samples"
        )
    else:
        header = (
            "layer_type,layer_index,subset,probe_site,property,probe_type,"
            "balanced_accuracy,balanced_std,accuracy,std,"
            "majority_baseline,n_classes,n_samples"
        )
    with open(path_a, "w") as f:
        f.write(header + "\n")
        for row in rows_a:
            if args.control:
                f.write(
                    f"{row['layer_type']},{row['layer_index']},{row['subset']},"
                    f"{row['probe_site']},{row['property']},{row['probe_type']},"
                    f"{row['accuracy']:.6f},{row['std']:.6f},"
                    f"{row['control_accuracy']:.6f},{row['control_std']:.6f},"
                    f"{row['selectivity']:.6f},"
                    f"{row['control_mode']},{row['n_controls']},"
                    f"{row['majority_baseline']:.6f},{row['n_classes']},{row['n_samples']}\n"
                )
            else:
                f.write(
                    f"{row['layer_type']},{row['layer_index']},{row['subset']},"
                    f"{row['probe_site']},{row['property']},{row['probe_type']},"
                    f"{row['balanced_accuracy']:.6f},{row['balanced_std']:.6f},"
                    f"{row['accuracy']:.6f},{row['std']:.6f},"
                    f"{row['majority_baseline']:.6f},{row['n_classes']},{row['n_samples']}\n"
                )
    logger.info("Saved %d analysis-A rows to %s", len(rows_a), path_a)

    # Analysis B CSV: identical schema to run_probes_lnl_within_stemfinal.py.
    with open(path_b, "w") as f:
        f.write(",".join(WITHIN_CSV_COLUMNS) + "\n")
        for row in rows_b:
            f.write(",".join(str(row[c]) for c in WITHIN_CSV_COLUMNS) + "\n")
    logger.info("Saved %d analysis-B rows to %s", len(rows_b), path_b)
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
