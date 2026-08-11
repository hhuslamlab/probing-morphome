"""Surface-baseline machinery and the reconstructed feature_engineering module."""

import os
import sys

import numpy as np
import pytest

from probing import FEATURE_INFORMED_ROOT
from probing.run_ngram_baselines import NgramLMClassifier, build_ngram_classifier

_scripts_dir = os.path.join(FEATURE_INFORMED_ROOT, "scripts")
if os.path.isdir(_scripts_dir) and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


def _have_feature_engineering():
    try:
        import feature_engineering  # noqa: F401
        return True
    except ImportError:
        return False


# feature_engineering ships with the model-training repo; skip its tests
# when that repo is not available.
needs_feature_engineering = pytest.mark.skipif(
    not _have_feature_engineering(),
    reason="feature_informed repo not available (set FEATURE_INFORMED_ROOT)",
)


class TestNGramClassifierTokenization:
    def test_phonemes_are_word_level_tokens(self):
        # Multi-codepoint IPA symbols must survive as single tokens; a
        # character analyzer would split them.
        clf = build_ngram_classifier(1, {"probe": {"linear": {"C": 1.0, "solver": "lbfgs",
                                                             "max_iter": 100},
                                                  "random_seed": 0}})
        vec = clf.named_steps["vec"]
        vec.fit(["t͡ʃ a", "a b"])
        assert "t͡ʃ" in vec.vocabulary_

class TestClassConditionalLM:
    def test_distinguishes_distinct_string_distributions(self):
        lm = NgramLMClassifier(order=2)
        a = ["a b a b", "a b a", "b a b"] * 10
        b = ["c d c d", "c d c", "d c d"] * 10
        X = np.array(a + b, dtype=object)
        y = np.array([0] * len(a) + [1] * len(b))
        lm.fit(X, y)
        pred = lm.predict(np.array(["a b a b a", "d c d c"], dtype=object))
        assert pred.tolist() == [0, 1]

    def test_uniform_prior_blocks_majority_default(self):
        # 90/10 skew with IDENTICAL string distributions: a prior-free
        # likelihood ratio should NOT collapse to the majority class.
        rng = np.random.RandomState(0)
        strings = [" ".join(rng.choice(list("abcd"), 5)) for _ in range(200)]
        X = np.array(strings, dtype=object)
        y = np.array([0] * 180 + [1] * 20)
        lm = NgramLMClassifier(order=2)
        lm.fit(X, y)
        pred = lm.predict(X)
        # both classes must actually be predicted
        assert set(pred.tolist()) == {0, 1}
        assert pred.mean() > 0.2  # far from the 0.0 a majority-guesser gives


@needs_feature_engineering
class TestReconstructedFeatureEngineering:
    def test_one_hot_layout_matches_pickled_convention(self):
        from feature_engineering import GrammaticalFeatureExtractor
        fx = GrammaticalFeatureExtractor()
        assert fx.feature_to_idx == {"IND": 0, "SBJV": 1, "1": 2, "2": 3,
                                     "3": 4, "SG": 5, "PL": 6}
        assert fx.get_feature_dim() == 7

    def test_tag_parsing_and_one_hot(self):
        from feature_engineering import GrammaticalFeatureExtractor
        fx = GrammaticalFeatureExtractor()
        feats = fx.extract_features_from_tag("<V;SBJV;PRS;1;PL>")
        assert feats == ["SBJV", "1", "PL"]
        vec = fx.create_one_hot_vector(feats)
        assert vec.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_feature_embedding_is_pure_linear(self):
        import torch
        from feature_engineering import FeatureEmbedding
        fe = FeatureEmbedding(7, 4)
        x = torch.zeros(7)
        x[1] = 1.0
        out = fe(x)
        expected = fe.linear.weight[:, 1] + fe.linear.bias
        assert torch.allclose(out, expected)
