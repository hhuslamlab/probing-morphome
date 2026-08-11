"""Content-position masks for leakage-free mean pooling of probe inputs.

The encoder input layout (TagInBracketsDataLoader._iter_helper) is

    BOS, tag1, #, tag2, #, ..., #, char1, char2, #, charN, EOS

so the morphological *tags* are front-loaded tokens.  Several probe targets
(mood/tense/person/number, src1_*/src2_*) are read straight off those tags, so
mean-pooling over the *padding* mask (every non-pad position, tags included)
lets a linear probe recover the label from the input — inflating accuracy.

A *content* mask keeps only the character positions: it drops BOS/EOS/UNK, the
morphological tags, and the ``#`` separators on the encoder side, and BOS/EOS/UNK
on the decoder side (the target sequence is ``BOS, char1..charN, EOS`` with no
tags).  The tag boundary follows the vocab construction in
``dataloader.py`` (``source = [PAD,BOS,EOS,UNK] + chars + tags``): a source id is
a tag iff ``id >= source_vocab_size - nb_attr``.
"""

import os

import torch

# Mirror the fixed special-token ids in dataloader.py (PAD=0, BOS=1, EOS=2, UNK=3).
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
_SPECIAL_IDS = (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)


def build_content_mask(tokens, side, source_vocab_size=None, nb_attr=None, sep_idx=None):
    """Build a content-position mask from a token-id tensor.

    Args:
        tokens: LongTensor [seq_len, batch] of source (encoder) or target
            (decoder) token ids, in the same orientation the model receives.
        side: "encoder" or "decoder".
        source_vocab_size: len(source vocab); required for the encoder.
        nb_attr: number of tag (attribute) tokens; required for the encoder.
        sep_idx: token id of the ``#`` separator; required for the encoder
            (pass ``data.source_c2i.get('#')``; may be None if absent).

    Returns:
        BoolTensor [batch, seq_len] (transposed to match the padding masks saved
        by extract_representations.py), True at content (character) positions.
    """
    # [batch, seq_len] so the result lines up with the saved padding masks.
    ids = tokens.transpose(0, 1).long()

    # Drop PAD/BOS/EOS/UNK on both sides.
    keep = torch.ones_like(ids, dtype=torch.bool)
    for special in _SPECIAL_IDS:
        keep &= ids != special

    if side == "encoder":
        assert source_vocab_size is not None and nb_attr is not None, (
            "source_vocab_size and nb_attr are required for the encoder content mask"
        )
        # Tags occupy the top nb_attr ids; characters are everything below.
        keep &= ids < (source_vocab_size - nb_attr)
        if sep_idx is not None:
            keep &= ids != sep_idx
    elif side != "decoder":
        raise ValueError(f"Unknown side: {side!r} (expected 'encoder' or 'decoder')")

    return keep


def build_tag_mask(tokens, source_vocab_size, nb_attr):
    """Build a TAG-position mask (encoder side only): True at morphological-tag
    tokens, False everywhere else (characters, separators, BOS/EOS/UNK/PAD).

    The complement of the character region used by ``build_content_mask``: tags
    occupy the top ``nb_attr`` source ids (``id >= source_vocab_size - nb_attr``).
    Used to probe what the tag positions themselves encode — for the
    feature_geometric model these positions carry the geometric FeatureEmbedding
    of mood/person/number, so this isolates that representation.
    """
    ids = tokens.transpose(0, 1).long()  # [batch, seq_len]
    keep = ids >= (source_vocab_size - nb_attr)
    for special in _SPECIAL_IDS:
        keep &= ids != special
    return keep


def mean_pool(representations, mask):
    """Mean-pool representations over sequence length, respecting the mask.

    Args:
        representations: [n_samples, seq_len, embed_dim]
        mask: [n_samples, seq_len] (bool, True = pool this position)

    Returns:
        Pooled representations: [n_samples, embed_dim]. Rows with no True
        position get a zero vector (lengths clamped to 1).
    """
    mask_float = mask.unsqueeze(-1).float()  # [n_samples, seq_len, 1]
    summed = (representations * mask_float).sum(dim=1)  # [n_samples, embed_dim]
    lengths = mask_float.sum(dim=1).clamp(min=1)  # [n_samples, 1]
    return summed / lengths


def last_pool(representations, mask):
    """Take the representation at the LAST True position of each row.

    For inflection features borne by the word-final suffix (and decoder layers
    that generate the form left-to-right), the final content token is a more
    targeted readout than a mean over the whole word.  Rows with no True
    position fall back to position 0.

    Args:
        representations: [n_samples, seq_len, embed_dim]
        mask: [n_samples, seq_len] (bool, True = candidate position)

    Returns:
        Pooled representations: [n_samples, embed_dim].
    """
    n, seq_len = representations.shape[0], representations.shape[1]
    ar = torch.arange(seq_len).view(1, -1)
    masked_idx = torch.where(mask.bool(), ar, torch.full_like(ar, -1))
    last_idx = masked_idx.max(dim=1).values.clamp(min=0)  # [n]
    return representations[torch.arange(n), last_idx]


def pool_reps(representations, mask, pool_positions):
    """Reduce per-position representations to one vector per example.

    'content', 'all', and 'tags' mean-pool over the mask (the mask itself selects
    which positions — content-only, every valid position, or tag positions only);
    'last' takes the last masked position.
    """
    if pool_positions == "last":
        return last_pool(representations, mask)
    return mean_pool(representations, mask)


def load_pool_mask(reps_dir, layer_type, layer_index, pool_positions="content", logger=None):
    """Load the mask to feed to the pooler for one layer.

    Args:
        reps_dir: directory holding ``<layer_type>_layer_<i>.pt`` and masks.
        layer_type: "encoder" or "decoder".
        layer_index: layer index.
        pool_positions: "content" (default) / "last" load ``<L>_content_mask.pt``
            and fall back to the padding mask with a loud warning if it is
            absent; "all" always loads the padding mask ``<L>_mask.pt``
            (reproduces the old, leaky numbers for A/B comparison).
        logger: optional logger for the fallback warning.

    Returns:
        BoolTensor [n_samples, seq_len], or None if no mask file is found.
    """
    base = f"{layer_type}_layer_{layer_index}"
    padding_path = os.path.join(reps_dir, f"{base}_mask.pt")
    content_path = os.path.join(reps_dir, f"{base}_content_mask.pt")

    if pool_positions == "all":
        if not os.path.exists(padding_path):
            return None
        return torch.load(padding_path, weights_only=False)

    if pool_positions == "tags":
        # Tag positions exist on the encoder side only; decoder has no tag mask,
        # so this returns None there and the caller skips the (meaningless) layer.
        tag_path = os.path.join(reps_dir, f"{base}_tag_mask.pt")
        if not os.path.exists(tag_path):
            return None
        return torch.load(tag_path, weights_only=False)

    if pool_positions not in ("content", "last"):
        raise ValueError(
            f"Unknown pool_positions: {pool_positions!r} "
            "(expected 'content', 'last', 'all', or 'tags')"
        )

    if os.path.exists(content_path):
        return torch.load(content_path, weights_only=False)

    if os.path.exists(padding_path):
        msg = (
            "No content mask at %s — falling back to the padding mask, which pools "
            "over tag tokens and LEAKS the probe label. Re-extract the "
            "representations to regenerate content masks." % content_path
        )
        if logger is not None:
            logger.warning(msg)
        else:
            import warnings

            warnings.warn(msg, stacklevel=2)
        return torch.load(padding_path, weights_only=False)

    return None
