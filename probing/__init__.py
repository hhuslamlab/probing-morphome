"""Probing pipeline for analyzing internal model representations."""

import os

# Root of the feature_informed repo, which owns the model source code
# (scripts/transformer.py, dataloader.py, ...), the trained checkpoints and the
# raw train/test data. Probing outputs live in THIS repo under data/.
FEATURE_INFORMED_ROOT = os.path.expanduser(
    os.environ.get("FEATURE_INFORMED_ROOT", "~/projects/research/feature_informed")
)

MODEL_TYPES = (
    "vanilla",
    "character_separated",
    "feature_invariant",
    "independent_feature",
    "feature_geometric",
)
SPLITS = ("10L_90NL", "50L_50NL", "90L_10NL")
PROPERTIES = (
    "tense", "mood", "person", "number",
    "l_shaped", "diphthongization", "alternation_class", "in_l_cell", "conjugation",
    "src1_mood", "src2_mood", "src1_person", "src2_person",
    "stem_final", "stem_final_match",
)
RUNS = (1, 2, 3, 4)

# Canonical reduced probe set for the 10L_90NL cross-architecture analysis.
#
# A run id is ``X_Y`` where X (1-3) selects the data split / held-out TEST SET and
# Y (1-4) is the training seed. All 4 seeds within a run share the SAME test set
# (MD5-identical) and the SAME global L/NL ``shape_info`` labels, so seeds are
# replicates while runs are the scientifically meaningful axis. To probe
# generalization across the 3 distinct test sets we keep one representative
# seed per run — seed 2, the seed available for every architecture — giving
# 3 runs x 5 architectures = 15 models instead of the full 12 x 5 = 60.
PROBE_RUNS_10L_90NL = ("1_2", "2_2", "3_2")

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_SKIPPED = 2

# Panphon subphonemic feature names used for probing (24 features).
# These correspond to the columns of panphon.FeatureTable().names.
SUBPHONEMIC_FEATURES = (
    "syl", "son", "cons", "cont", "delrel", "lat", "nas",
    "strid", "voi", "sg", "cg", "ant", "cor", "distr",
    "lab", "hi", "lo", "back", "round", "velaric",
    "tense", "long", "hitone", "hireg",
)
