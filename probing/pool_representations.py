"""Pool per-layer representations once and cache the result locally.

Reduces each [n_samples, seq_len, embed_dim] layer tensor on the slow external
drive to a pooled [n_samples, embed_dim] .npy under <cache-dir>, so downstream
probing never touches the drive again. Chunked pooling is bit-identical to one
unchunked pool_reps call; writes are atomic and per-layer idempotent.

Usage:
  pool_representations.py --model-type TYPE --split SPLIT --run RUN
                          [--representations-dir DIR] [--cache-dir DIR]
                          [--pool-positions POS] [--chunk-size N]
  pool_representations.py (-h | --help)

Options:
  --model-type TYPE          Architecture (one of the five MODEL_TYPES).
  --split SPLIT              Data split, e.g. 10L_90NL.
  --run RUN                  Run identifier, e.g. 1_2.
  --representations-dir DIR  Directory with extracted representations
                             [default: data/probing/representations].
  --cache-dir DIR            Local directory for pooled .npy files
                             [default: data/probing/pooled_cache].
  --pool-positions POS       Pooling readout (content, all, last); must match
                             the probe script's --pool-positions [default: content].
  --chunk-size N             Rows pooled per chunk (caps the mean_pool
                             broadcast temp) [default: 8192].
"""

import json
import os
import shutil

import numpy as np
import torch

from probing import MODEL_TYPES, SPLITS
from probing.utils.cli import parse
from probing.utils.content_mask import load_pool_mask, pool_reps


def parse_args():
    return parse(__doc__,
                 types=dict(chunk_size=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS,
                              pool_positions=("content", "all", "last")))


def pool_in_chunks(reps, mask, pool_positions, chunk_size):
    """Row-chunked pool_reps; bit-identical to one unchunked call."""
    chunks = []
    for start in range(0, reps.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append(pool_reps(reps[start:end], mask[start:end], pool_positions))
    return torch.cat(chunks, dim=0)


if __name__ == "__main__":
    args = parse_args()

    reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Representations not found: {reps_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)
    n_encoder = metadata["n_encoder_layers"]
    n_decoder = metadata["n_decoder_layers"]

    cache_dir = os.path.join(args.cache_dir, args.model_type, f"{args.split}_{args.run}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_metadata_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(cache_metadata_path):
        shutil.copy2(metadata_path, cache_metadata_path)

    layers = [("encoder", i) for i in range(n_encoder)] + [("decoder", i) for i in range(n_decoder)]

    n_pooled = 0
    n_skipped = 0
    for layer_type, layer_index in layers:
        base = f"{layer_type}_layer_{layer_index}"
        out_path = os.path.join(cache_dir, f"{base}_{args.pool_positions}.npy")
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        rep_path = os.path.join(reps_dir, f"{base}.pt")
        if not os.path.exists(rep_path):
            raise FileNotFoundError(f"Missing rep file: {rep_path}")
        mask = load_pool_mask(reps_dir, layer_type, layer_index, args.pool_positions)
        if mask is None:
            raise FileNotFoundError(f"Missing mask file for {base} in {reps_dir}")

        reps = torch.load(rep_path, weights_only=False)
        pooled = pool_in_chunks(reps, mask, args.pool_positions, args.chunk_size).numpy()
        del reps

        # PID-unique tmp name so concurrent instances pooling the same layer
        # cannot race each other's rename; os.replace stays last-writer-wins.
        tmp_path = f"{out_path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, pooled)
        os.replace(tmp_path, out_path)
        n_pooled += 1
