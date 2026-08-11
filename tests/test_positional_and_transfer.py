"""Stem-final position location, transfer-probe fold logic, and the
structure-preserving control shuffle."""

import numpy as np

from probing.pool_stemfinal_position import form_tokens, stem_final_token_index
from probing.run_probes_stemfinal_lnl import shuffle_labels_by_group
from probing.run_transfer_probe import transfer_score


class TestStemFinalIndex:
    def test_simple_form(self):
        # salgo -> stem salg- (suffix -o), stem-final consonant ɡ at index 3
        assert stem_final_token_index(["s", "a", "l", "ɡ", "o"]) == 3

    def test_stress_mark_is_skipped(self):
        # sˈalɡo: stress token must never be selected
        toks = ["s", "ˈ", "a", "l", "ɡ", "o"]
        assert toks[stem_final_token_index(toks)] == "ɡ"

    def test_suffixless_form_falls_back_to_whole_form(self):
        # no known suffix strips -> stem = whole form, last consonant chosen
        toks = ["b", "u", "t"]
        assert toks[stem_final_token_index(toks)] == "t"

    def test_vowel_final_stem_falls_back_to_stem_end(self):
        # all-vowel stem: no consonant to pick, return last stem token
        idx = stem_final_token_index(["a", "e", "o"])
        assert idx in (0, 1)  # stem end after stripping suffix 'o' / fallback

    def test_form_tokens_stop_at_tag(self):
        seg = "b u s o V IND PRS 1 SG"
        assert form_tokens(seg) == ["b", "u", "s", "o"]


class TestTransferScore:
    def _data(self, transferable, n_groups=30, per_group=8, seed=0):
        rng = np.random.RandomState(seed)
        groups = np.repeat(np.arange(n_groups), per_group)
        y = np.array([g % 2 for g in groups])
        n = len(y)
        # subset assignment: half of each group's samples in src, half in tgt
        src = np.tile(np.array([True] * 4 + [False] * 4), n_groups)
        tgt = ~src
        X = rng.randn(n, 6)
        if transferable:
            X[:, 0] += 2.5 * (2 * y - 1)  # same code in both subsets
        else:
            X[src, 1] += 2.5 * (2 * y[src] - 1)   # src-only code
        return X, y, groups, src, tgt

    def _probe(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return lambda: Pipeline([("s", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=200))])

    def test_shared_code_transfers(self):
        X, y, groups, src, tgt = self._data(transferable=True)
        bal = transfer_score(X, y, groups, src, tgt, self._probe(), 5, 42)
        assert bal > 0.9

    def test_subset_specific_code_does_not_transfer(self):
        X, y, groups, src, tgt = self._data(transferable=False)
        bal = transfer_score(X, y, groups, src, tgt, self._probe(), 5, 42)
        assert 0.35 < bal < 0.65

    def test_lemma_memorization_cannot_leak(self):
        # Labels decodable only via group identity (a lookup table): with
        # lemma-disjoint folds transfer must stay at chance.
        rng = np.random.RandomState(1)
        groups = np.repeat(np.arange(30), 8)
        y = np.array([g % 2 for g in groups])
        X = np.eye(30)[groups] + 0.01 * rng.randn(len(y), 30)  # group one-hots
        src = np.tile(np.array([True] * 4 + [False] * 4), 30)
        bal = transfer_score(X, y, groups, src, ~src, self._probe(), 5, 42)
        assert 0.3 < bal < 0.7


class TestGroupShuffleControl:
    def test_each_group_keeps_one_consistent_label(self):
        rng = np.random.RandomState(0)
        groups = np.repeat(np.arange(10), 5)
        y = np.array([g % 2 for g in groups])
        y_c = shuffle_labels_by_group(y, groups, rng)
        for g in range(10):
            assert len(set(y_c[groups == g])) == 1

    def test_label_multiset_over_groups_is_preserved(self):
        rng = np.random.RandomState(0)
        groups = np.repeat(np.arange(10), 5)
        y = np.array([g % 2 for g in groups])
        y_c = shuffle_labels_by_group(y, groups, rng)
        orig = sorted(y[::5].tolist())
        ctrl = sorted(y_c[::5].tolist())
        assert orig == ctrl


class TestPreAlternantIndex:
    def test_predicting_state_precedes_alternant(self):
        # Under teacher forcing the state over input char k predicts char
        # k+1, so the state predicting the stem-final consonant sits at
        # content position sf-1 and has not yet seen it. salgas: tokens
        # s a l ɡ a s, stem salg-, sf index 3 -> pre-alternant position 2.
        toks = ["s", "a", "l", "ɡ", "a", "s"]
        sf = stem_final_token_index(toks)
        assert toks[sf] == "ɡ"
        assert sf - 1 == 2 and toks[sf - 1] == "l"  # predictor state's input

    def test_form_initial_stem_final_is_invalid(self):
        # A form whose stem-final consonant is its first character has no
        # preceding content position; pooling marks such rows invalid.
        toks = ["d", "a"]
        sf = stem_final_token_index(toks)
        assert sf == 0  # no position sf-1 exists in the content mask
