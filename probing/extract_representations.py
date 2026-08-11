"""Extract hidden state representations from a single model configuration.

Usage:
  extract_representations.py --model-type TYPE --split SPLIT --run RUN --checkpoint FILE
                             [--data-dir DIR] [--output-dir DIR] [--batch-size N]
                             [--baseline MODE] [--baseline-seed N]
  extract_representations.py (-h | --help)

Options:
  --model-type TYPE   Architecture (one of the five MODEL_TYPES).
  --split SPLIT       Data split, e.g. 90L_10NL.
  --run RUN           Run identifier, e.g. 1_1.
  --checkpoint FILE   Path to the model checkpoint file.
  --data-dir DIR      Base data directory [default: FEATURE_INFORMED_DATA].
  --output-dir DIR    Output directory for representations
                      [default: data/probing/representations].
  --batch-size N      Batch size for inference [default: 32].
  --baseline MODE     Baseline mode for selectivity: none, random_init
                      (re-initialise all model weights, i.e. representations
                      from an untrained encoder of identical architecture), or
                      scrambled_input (permute non-pad source positions per
                      sample before the forward pass). Outputs go to a
                      baseline-suffixed directory so they do not overwrite
                      real extractions [default: none].
  --baseline-seed N   Seed for baseline randomisation (weight reset / input
                      scramble) [default: 1337].
"""

import json
import os
import sys

import numpy as np
import torch

# Add scripts/ to sys.path so torch.load can unpickle model classes
# (models were saved when 'transformer', 'binary_feature_transformer' etc. were top-level imports)
from probing import FEATURE_INFORMED_ROOT

_scripts_dir = os.path.join(FEATURE_INFORMED_ROOT, "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from probing import MODEL_TYPES, SPLITS, EXIT_SUCCESS, EXIT_ERROR, EXIT_SKIPPED
from probing.utils.cli import parse, standard_sentinels
from probing.utils.hooks import HookManager
from probing.utils.content_mask import build_content_mask


def parse_args():
    return parse(__doc__,
                 types=dict(batch_size=int, baseline_seed=int),
                 choices=dict(model_type=MODEL_TYPES, split=SPLITS,
                              baseline=("none", "random_init", "scrambled_input")),
                 sentinels=standard_sentinels())


def reset_model_weights(model, seed):
    """Deterministically re-initialise every parameter in place (reset_parameters where available, else a small normal draw)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    touched_param_ids = set()
    for module in model.modules():
        if module is model:
            continue
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            try:
                module.reset_parameters()
                for p in module.parameters(recurse=False):
                    touched_param_ids.add(id(p))
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: reset_parameters failed on {module.__class__.__name__}: {e} — will fall back to normal init", file=sys.stderr)

    fallback_count = 0
    for name, param in model.named_parameters():
        if id(param) in touched_param_ids:
            continue
        with torch.no_grad():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)
        fallback_count += 1
    if fallback_count:
        pass


def scramble_src(src, src_mask, global_offset, base_seed):
    """Permute non-pad source positions per sample, deterministically (src and src_mask are [seq_len, batch_size])."""
    seq_len, batch_size = src.shape
    out = src.clone()
    for b in range(batch_size):
        valid_pos = (src_mask[:, b] > 0).nonzero(as_tuple=True)[0]
        if valid_pos.numel() < 2:
            continue
        rng = np.random.RandomState(base_seed + global_offset + b)
        perm = rng.permutation(valid_pos.numel())
        valid_pos_cpu = valid_pos.detach().cpu()
        perm_pos = valid_pos_cpu[perm].to(valid_pos.device)
        out[valid_pos, b] = src[perm_pos, b]
    return out


def _baseline_output_dir(base_dir, baseline):
    if baseline == "none":
        return base_dir
    return f"{base_dir}__{baseline}"


def get_data_files(data_dir, split, run):
    """Return train/dev/test paths: {data_dir}/{split}/{subset}/run{run_num}/{subset}.{split}_{run}.src/.tgt."""
    run_num = run.split("_")[0]
    model_name = f"{split}_{run}"

    result = {}
    for subset in ("train", "dev", "test"):
        src = os.path.join(data_dir, split, subset, f"run{run_num}", f"{subset}.{model_name}.src")
        tgt = os.path.join(data_dir, split, subset, f"run{run_num}", f"{subset}.{model_name}.tgt")
        result[subset] = [src, tgt]

    return result


def pad_and_concatenate(tensor_list, pad_dim=1):
    """Pad tensors to the same size along pad_dim, then concatenate along dim 0."""
    max_size = max(t.shape[pad_dim] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_amount = max_size - t.shape[pad_dim]
        if pad_amount > 0:
            pad_shape = list(t.shape)
            pad_shape[pad_dim] = pad_amount
            padding = torch.zeros(pad_shape, dtype=t.dtype)
            t = torch.cat([t, padding], dim=pad_dim)
        padded.append(t)
    return torch.cat(padded, dim=0)


if __name__ == "__main__":
    args = parse_args()

    # Check idempotency: if output already exists, skip.  The output root is
    # suffixed when running in a baseline mode so we never overwrite the real
    # extraction with random-init or scrambled-input activations.
    effective_output_dir = _baseline_output_dir(args.output_dir, args.baseline)
    output_path = os.path.join(effective_output_dir, args.model_type, f"{args.split}_{args.run}")
    check_file = os.path.join(output_path, "encoder_layer_0.pt")
    if os.path.exists(check_file):
        print(f"Skipping {args.model_type}/{args.split}_{args.run} (baseline={args.baseline}) -- already extracted")
        sys.exit(EXIT_SKIPPED)

    files = get_data_files(args.data_dir, args.split, args.run)
    for subset, paths in files.items():
        for path in paths:
            if not os.path.exists(path):
                sys.exit(f"Missing {subset} file: {path}")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # Import model classes so torch.load can unpickle them.  transformer is
    # always required: vanilla / character_separated use Transformer directly,
    # and feature_invariant is TagTransformer (defined there too). The other
    # two modules are only needed to unpickle their own architectures; import
    # them best-effort so a missing optional dependency (e.g.
    # feature_engineering) doesn't block unrelated extractions.
    import transformer  # noqa: F401
    for _optional_model_module in ("binary_feature_transformer", "independent_feature_transformer"):
        try:
            __import__(_optional_model_module)
        except ImportError as e:
            print(f"WARNING: Optional model module {_optional_model_module} unavailable ({e}); checkpoints of that architecture cannot be unpickled.", file=sys.stderr)

    # Some checkpoints (e.g. several feature_invariant 50L/90L runs) were pickled
    # from a repo layout where these modules lived under a top-level `src`
    # package, so their globals are qualified as `src.transformer.TagTransformer`
    # etc. Alias `src.<mod>` to the already-imported flat module so torch.load can
    # resolve them. Best-effort: only modules that imported successfully are aliased.
    import types
    _src_pkg = sys.modules.get("src")
    if _src_pkg is None:
        _src_pkg = types.ModuleType("src")
        _src_pkg.__path__ = []  # mark as a package
        sys.modules["src"] = _src_pkg
    for _flat in ("transformer", "binary_feature_transformer",
                  "independent_feature_transformer", "dataloader"):
        if _flat in sys.modules:
            sys.modules.setdefault(f"src.{_flat}", sys.modules[_flat])

    # All five architectures use TagInBracketsDataLoader with the
    # taginbrackets dataset; they differ only in how features are embedded.
    from dataloader import TagInBracketsDataLoader

    data = TagInBracketsDataLoader(
        files["train"], files["dev"], files["test"], shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    if args.baseline == "random_init":
        reset_model_weights(model, args.baseline_seed)
        model.eval()
    elif args.baseline == "scrambled_input":
        pass

    hook_manager = HookManager()
    hook_manager.register_hooks(model)

    os.makedirs(output_path, exist_ok=True)

    # All layer activations are accumulated in RAM and written out at the end.
    # Peak usage is ~11 GB for the 43k-sample test set (8 layers x
    # [n, seq, 256] float32).
    # Keyed by (layer_type, layer_index); values are lists of [batch_size, seq_len, embed_dim] tensors
    batch_reps = {}
    batch_src_masks = []
    batch_trg_masks = []
    # Content masks (character positions only) for leakage-free pooling.
    batch_src_content_masks = []
    batch_trg_content_masks = []
    sep_idx = data.source_c2i.get("#")

    global_offset = 0
    with torch.no_grad():
        for src, src_mask, trg, trg_mask in data.test_batch_sample(args.batch_size):
            # src_mask: [seq_len, batch_size], float, 1=valid 0=pad
            if args.baseline == "scrambled_input":
                src = scramble_src(
                    src, src_mask, global_offset=global_offset, base_seed=args.baseline_seed,
                )

            batch_src_masks.append(src_mask.transpose(0, 1).bool())  # [batch, src_seq_len]
            batch_trg_masks.append(trg_mask.transpose(0, 1).bool())  # [batch, trg_seq_len]

            # Content masks: derived from the (possibly scrambled) tokens, so they
            # stay aligned with the reps even under baseline=scrambled_input.
            batch_src_content_masks.append(
                build_content_mask(
                    src, "encoder",
                    source_vocab_size=data.source_vocab_size,
                    nb_attr=data.nb_attr,
                    sep_idx=sep_idx,
                )
            )
            batch_trg_content_masks.append(build_content_mask(trg, "decoder"))

            # Teacher-forced forward pass; model handles device transfer internally
            _ = model(src, src_mask, trg, trg_mask)
            global_offset += src.shape[1]

            reps = hook_manager.get_representations()
            for key, tensors in reps.items():
                if key not in batch_reps:
                    batch_reps[key] = []
                # Each tensor is transposed from [seq_len, batch_size, embed_dim] to [batch_size, seq_len, embed_dim]
                for t in tensors:
                    batch_reps[key].append(t.transpose(0, 1))

            hook_manager.clear()

    hook_manager.remove_hooks()

    if not batch_reps:
        sys.exit("No representations collected -- check model and data")

    # Masks are tiny (no embed dim) so they stay in RAM; pad to global max seq_len.
    src_masks = pad_and_concatenate(batch_src_masks, pad_dim=1)  # [n_samples, max_src_len]
    trg_masks = pad_and_concatenate(batch_trg_masks, pad_dim=1)  # [n_samples, max_trg_len]
    # Same padding/concat path → content masks are row/col aligned with the reps.
    src_content_masks = pad_and_concatenate(batch_src_content_masks, pad_dim=1)
    trg_content_masks = pad_and_concatenate(batch_trg_content_masks, pad_dim=1)

    n_samples = src_masks.shape[0]
    embed_dim = None
    n_encoder_layers = 0
    n_decoder_layers = 0

    for layer_type, layer_index in sorted(batch_reps):
        # Free each layer's batch list as it is assembled and saved.
        tensors = batch_reps.pop((layer_type, layer_index))
        rep = pad_and_concatenate(tensors, pad_dim=1)  # [n_samples, seq_len, embed_dim]
        embed_dim = rep.shape[2]

        filename = f"{layer_type}_layer_{layer_index}.pt"
        torch.save(rep, os.path.join(output_path, filename))

        mask = src_masks if layer_type == "encoder" else trg_masks
        mask_filename = f"{layer_type}_layer_{layer_index}_mask.pt"
        torch.save(mask, os.path.join(output_path, mask_filename))

        content_mask = src_content_masks if layer_type == "encoder" else trg_content_masks
        content_mask_filename = f"{layer_type}_layer_{layer_index}_content_mask.pt"
        torch.save(content_mask, os.path.join(output_path, content_mask_filename))


        del rep, tensors

        if layer_type == "encoder":
            n_encoder_layers += 1
        else:
            n_decoder_layers += 1

    metadata = {
        "embed_dim": embed_dim,
        "n_encoder_layers": n_encoder_layers,
        "n_decoder_layers": n_decoder_layers,
        "n_samples": n_samples,
        "src_seq_len": int(src_masks.shape[1]),
        "trg_seq_len": int(trg_masks.shape[1]),
        "baseline": args.baseline,
        "baseline_seed": args.baseline_seed if args.baseline != "none" else None,
        "mean_src_content_tokens": float(src_content_masks.float().sum(dim=1).mean()),
        "mean_trg_content_tokens": float(trg_content_masks.float().sum(dim=1).mean()),
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    sys.exit(EXIT_SUCCESS)
