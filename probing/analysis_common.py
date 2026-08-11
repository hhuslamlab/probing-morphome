"""Shared plumbing for the analysis drivers (positional, transfer, structure).

Every driver follows the same shape: parse args, skip if its output CSV
exists, load labels and lemma groups, iterate the 8 layers, compute rows,
write the CSV, exit 0/1/2. This module holds that shape so each driver
contains only the science.
"""

import logging
import os
import sys

import numpy as np

from probing import EXIT_SKIPPED
from probing.run_probes_stemfinal_lnl import (
    build_lemma_groups,
    build_property_labels,
    get_src_path,
    get_tgt_path,
)

# The 8 probed layers, in encoder-to-decoder order. layer_name() gives the
# short form used in every CSV, table, and figure ("enc0" .. "dec3").
LAYERS = [("encoder", i) for i in range(4)] + [("decoder", i) for i in range(4)]


def layer_name(layer_type, layer_index):
    return layer_type[:3] + str(layer_index)


def output_path_or_skip(output_dir, model_type, filename, logger):
    """Resolve <output-dir>/<model_type>/<filename>; exit 2 if it exists."""
    out_dir = os.path.join(output_dir, model_type)
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        logger.info("Skipping %s -- exists", out_path)
        sys.exit(EXIT_SKIPPED)
    return out_path


def load_labels_and_groups(data_dir, split, run):
    """Property labels dict + lemma group array for one (split, run)."""
    labels = build_property_labels(
        get_src_path(data_dir, split, run),
        get_tgt_path(data_dir, split, run),
        data_dir,
    )
    groups = np.asarray(build_lemma_groups(data_dir, split, run))
    return labels, groups


def load_layer(cache_dir, layer_type, layer_index, suffix="content"):
    """One pooled layer matrix [n, 256] from the local cache."""
    path = os.path.join(cache_dir, f"{layer_type}_layer_{layer_index}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pooled layer {path}")
    return np.load(path)


def write_rows(out_path, rows, logger):
    """Write list-of-dict rows as CSV (column order = first row's key order)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    logger.info("Saved %d rows to %s", len(rows), out_path)


def setup_logging(name):
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)
