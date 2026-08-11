"""Content-position masks for leakage-free mean pooling of probe inputs.

Encoder inputs front-load the morphological tag tokens, so pooling over the
padding mask (every non-pad position) lets a probe read labels straight off
the tag positions, inflating accuracy. The content mask keeps only character
positions, dropping tags, separators, and specials (a source id is a tag iff
id >= source_vocab_size - nb_attr, per the vocab layout in dataloader.py).
"""

import os

import torch

# Mirror the fixed special-token ids in dataloader.py (PAD=0, BOS=1, EOS=2, UNK=3).
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
_SPECIAL_IDS = (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)


def build_content_mask(tokens, side, source_vocab_size=None, nb_attr=None, sep_idx=None):
    """Content mask from token ids (tokens are [seq_len, batch]; returns bool [batch, seq_len])."""
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
    """Tag-position mask, encoder side only (tags occupy the top nb_attr source ids)."""
    ids = tokens.transpose(0, 1).long()  # [batch, seq_len]
    keep = ids >= (source_vocab_size - nb_attr)
    for special in _SPECIAL_IDS:
        keep &= ids != special
    return keep


def mean_pool(representations, mask):
    """Masked mean over sequence length (rows with no True position get zeros)."""
    mask_float = mask.unsqueeze(-1).float()  # [n_samples, seq_len, 1]
    summed = (representations * mask_float).sum(dim=1)  # [n_samples, embed_dim]
    lengths = mask_float.sum(dim=1).clamp(min=1)  # [n_samples, 1]
    return summed / lengths


def last_pool(representations, mask):
    """Representation at each row's last True position (empty rows fall back to position 0)."""
    n, seq_len = representations.shape[0], representations.shape[1]
    ar = torch.arange(seq_len).view(1, -1)
    masked_idx = torch.where(mask.bool(), ar, torch.full_like(ar, -1))
    last_idx = masked_idx.max(dim=1).values.clamp(min=0)  # [n]
    return representations[torch.arange(n), last_idx]


def pool_reps(representations, mask, pool_positions):
    """One vector per example ('last' takes the last masked position; others mean-pool)."""
    if pool_positions == "last":
        return last_pool(representations, mask)
    return mean_pool(representations, mask)


def load_pool_mask(reps_dir, layer_type, layer_index, pool_positions="content"):
    """Load one layer's pooling mask, or None ('content' falls back to the leaky padding mask with a warning)."""
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
        import warnings

        warnings.warn(
            f"No content mask at {content_path} — falling back to the padding "
            "mask, which pools over tag tokens and LEAKS the probe label. "
            "Re-extract the representations to regenerate content masks.",
            stacklevel=2,
        )
        return torch.load(padding_path, weights_only=False)

    return None
