"""Pool per-layer representations once and cache the result locally.

The full [n_samples, seq_len, embed_dim] representation tensors live on a slow
external drive (~0.8-1.7 GB per layer), but every probe script immediately
reduces them to a [n_samples, embed_dim] pooled matrix (~44 MB).  This script
does that reduction in one sequential pass per layer -- one read of each .pt
off the drive -- and saves the pooled float32 matrix as
``<cache-dir>/<model_type>/<split>_<run>/<layer_type>_layer_<i>_<pool_positions>.npy``
plus a copy of ``metadata.json``, so downstream probing never touches the
external drive again.

Pooling reuses load_pool_mask/pool_reps unchanged but runs in chunks along the
sample dimension to cap peak RAM (mean_pool materialises a full-size broadcast
temp otherwise); both poolers are row-independent, so chunking is bit-identical.
Writes are atomic (tmp file + os.replace) and per-layer idempotent, so an
interrupted run resumes where it left off.

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
import logging
import os
import shutil
import sys

import numpy as np
import torch

from probing import EXIT_ERROR, EXIT_SUCCESS, MODEL_TYPES, SPLITS
from probing.utils.cli import parse
from probing.utils.content_mask import load_pool_mask, pool_reps

logger = logging.getLogger(__name__)


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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    reps_dir = os.path.join(args.representations_dir, args.model_type, f"{args.split}_{args.run}")
    metadata_path = os.path.join(reps_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error("Representations not found: %s", reps_dir)
        sys.exit(EXIT_ERROR)

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
            logger.error("Missing rep file: %s", rep_path)
            sys.exit(EXIT_ERROR)
        mask = load_pool_mask(reps_dir, layer_type, layer_index, args.pool_positions, logger=logger)
        if mask is None:
            logger.error("Missing mask file for %s in %s", base, reps_dir)
            sys.exit(EXIT_ERROR)

        reps = torch.load(rep_path, weights_only=False)
        pooled = pool_in_chunks(reps, mask, args.pool_positions, args.chunk_size).numpy()
        del reps

        # PID-unique tmp name so concurrent instances pooling the same layer
        # cannot race each other's rename; os.replace stays last-writer-wins.
        tmp_path = f"{out_path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, pooled)
        os.replace(tmp_path, out_path)
        n_pooled += 1
        logger.info("Pooled %s -> %s %s", base, out_path, pooled.shape)

    logger.info(
        "%s/%s_%s: pooled %d layers, %d already cached",
        args.model_type,
        args.split,
        args.run,
        n_pooled,
        n_skipped,
    )
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
