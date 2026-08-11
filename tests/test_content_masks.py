"""Content-mask correctness across architectures.

Locks in the char_sep tag handling: decomposed tag tokens (``V SBJV PRS 1
PL``) must be excluded from encoder content masks. A bracket-only check
(``<TAG>``) would never fire in the char_sep vocabulary and would silently
pool over tags for that architecture.
"""

import torch

from probing.extract_representations_char_sep import is_content_symbol
from probing.utils.content_mask import build_content_mask, pool_reps


class TestCharSepContentSymbol:
    def test_ipa_characters_are_content(self):
        for sym in ("a", "s", "ɲ", "ʐ", "ˈ"):  # incl. stress mark
            assert is_content_symbol(sym, "encoder")

    def test_decomposed_tag_tokens_are_not_content(self):
        # These would pass a bracket-only check.
        for sym in ("V", "IND", "SBJV", "PRS", "SG", "PL", "1", "2", "3"):
            assert not is_content_symbol(sym, "encoder")

    def test_bracketed_tags_and_separator_are_not_content(self):
        assert not is_content_symbol("<V;IND;PRS;1;SG>", "encoder")
        assert not is_content_symbol("#", "encoder")

    def test_specials_are_not_content_either_side(self):
        for sym in ("<s>", "<pad>", "</s>", "<unk>"):
            assert not is_content_symbol(sym, "encoder")
            assert not is_content_symbol(sym, "decoder")

    def test_decoder_keeps_every_non_special(self):
        assert is_content_symbol("a", "decoder")


class TestBuildContentMask:
    def test_encoder_drops_tags_separator_and_specials(self):
        # vocab: [PAD,BOS,EOS,UNK] + chars(4,5,6=sep) + tags(7,8) -> size 9, nb_attr 2
        # seq (len x batch=1): BOS tag7 sep6 char4 char5 EOS
        tokens = torch.tensor([[1], [7], [6], [4], [5], [2]])
        mask = build_content_mask(tokens, "encoder", source_vocab_size=9, nb_attr=2, sep_idx=6)
        assert mask.shape == (1, 6)
        assert mask[0].tolist() == [False, False, False, True, True, False]

    def test_pool_reps_mean_over_masked_positions_only(self):
        reps = torch.zeros(1, 4, 2)
        reps[0, 1] = torch.tensor([2.0, 4.0])
        reps[0, 2] = torch.tensor([4.0, 8.0])
        mask = torch.tensor([[False, True, True, False]])
        pooled = pool_reps(reps, mask, "content")
        assert torch.allclose(pooled[0], torch.tensor([3.0, 6.0]))

    def test_pool_reps_last_takes_final_true_position(self):
        reps = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        mask = torch.tensor([[True, True, True, False]])
        pooled = pool_reps(reps, mask, "last")
        assert torch.allclose(pooled[0], torch.tensor([4.0, 5.0]))
