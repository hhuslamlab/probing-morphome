"""Extract hidden-state representations from the char_sep (fairseq) models.

The char_sep checkpoints are fairseq ``transformer`` state_dicts, not the
pickled in-house ``transformer.Transformer`` objects, so this script rebuilds
that architecture in pure PyTorch (fairseq is deliberately not a dependency).
Output layout is identical to ``extract_representations.py``, so downstream
scripts run unchanged.  The fairseq dictionaries must match the checkpoint's
embedding sizes; a mismatch aborts rather than emit garbage vectors.

Usage:
  extract_representations_char_sep.py --checkpoint FILE --data-bin DIR
                                      [--model-type NAME] [--output-dir DIR]
                                      [--batch-size N]
                                      [--test-src FILE] [--test-tgt FILE]
  extract_representations_char_sep.py --all --data-root DIR
                                      [--checkpoint-root DIR] [--model-type NAME]
                                      [--output-dir DIR] [--batch-size N]
  extract_representations_char_sep.py (-h | --help)

Options:
  --checkpoint FILE      Path to a fairseq checkpoint .pt (single-model mode).
  --data-bin DIR         fairseq data dir with dict.<run>.{src,tgt}.txt.
  --test-src FILE        Raw whitespace-tokenised test .src
                         (default: <data-bin>/test.<run>.src).
  --test-tgt FILE        Raw whitespace-tokenised test .tgt
                         (default: <data-bin>/test.<run>.tgt).
  --all                  Iterate every *-models dir under --checkpoint-root,
                         skipping *_old / *_bk.
  --checkpoint-root DIR  Root of the char_sep checkpoint dirs
                         [default: FEATURE_INFORMED_CHAR_SEP_CKPTS].
  --data-root DIR        Root holding per-run fairseq data dirs <split>_<run>/.
  --model-type NAME      Output sub-directory name [default: char_sep].
  --output-dir DIR       Output root [default: data/probing/representations].
  --batch-size N         Batch size for inference [default: 32].
"""

import glob
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from torch import nn

from probing import EXIT_SUCCESS, EXIT_ERROR, EXIT_SKIPPED, FEATURE_INFORMED_ROOT
from probing.utils.cli import parse, standard_sentinels

# fairseq Dictionary fixed special-token ids (Dictionary.__init__ adds them in
# this order): <s>=bos, <pad>, </s>=eos, <unk>.
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>")


# fairseq dictionary + tokenisation
def load_fairseq_dict(path):
    """Load a fairseq ``dict.*.txt`` into (symbols, sym2idx); the four specials are implicit at indices 0 to 3."""
    symbols = list(_SPECIALS)
    sym2idx = {s: i for i, s in enumerate(symbols)}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # rsplit once: the symbol itself may contain spaces in principle,
            # the trailing field is the (unused) frequency count.
            sym, _count = line.rsplit(" ", 1)
            sym2idx[sym] = len(symbols)
            symbols.append(sym)
    return symbols, sym2idx


def encode_line(line, sym2idx, append_eos=True):
    """Whitespace-tokenise and numberise one line, fairseq-style (append eos)."""
    ids = [sym2idx.get(tok, UNK_IDX) for tok in line.split()]
    if append_eos:
        ids.append(EOS_IDX)
    return ids


def is_content_symbol(sym, side):
    """True if sym is an IPA character (encoder side also drops '#', uppercase/digit tag tokens, and '<'-prefixed tokens)."""
    if sym in _SPECIALS:
        return False
    if side == "encoder":
        return not (sym.startswith("<") or sym == "#" or sym.isupper() or sym.isdigit())
    return True


# pure-torch re-implementation of the fairseq transformer (arch=transformer)
def sinusoidal_table(num_embeddings, embedding_dim, padding_idx):
    """fairseq SinusoidalPositionalEmbedding.get_embedding (verbatim maths)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    emb[padding_idx, :] = 0
    return emb


def make_positions(tokens, padding_idx):
    """fairseq utils.make_positions: non-pad tokens get consecutive positions starting at padding_idx+1."""
    mask = tokens.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1300):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self._table = sinusoidal_table(init_size, embedding_dim, padding_idx)

    def forward(self, tokens):
        positions = make_positions(tokens, self.padding_idx)  # [B, T]
        max_pos = int(positions.max()) + 1
        if max_pos > self._table.size(0):
            self._table = sinusoidal_table(max_pos, self.embedding_dim, self.padding_idx)
        table = self._table.to(tokens.device)
        return table[positions]  # [B, T, C]


class MultiheadAttention(nn.Module):
    """fairseq MultiheadAttention (separate q/k/v projections, q scaled by head_dim**-0.5)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # query/key/value: [T, B, C]
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        def reshape(x, length):
            # reshape [length, B, C] to [B*H, length, head_dim]
            return (
                x.contiguous()
                .view(length, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        q = reshape(q, tgt_len)
        k = reshape(k, src_len)
        v = reshape(v, src_len)

        attn = torch.bmm(q, k.transpose(1, 2))  # [B*H, tgt_len, src_len]

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0)  # [1, tgt_len, src_len] broadcast

        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, tgt_len, src_len)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn = attn.view(bsz * self.num_heads, tgt_len, src_len)

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # [B*H, tgt_len, head_dim]
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN (normalize_before=True) fairseq encoder layer, relu activation."""

    def __init__(self, embed_dim, ffn_dim, heads):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, key_padding_mask=encoder_padding_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-LN fairseq decoder layer: causal self-attn, cross-attn, FFN."""

    def __init__(self, embed_dim, ffn_dim, heads):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn = MultiheadAttention(embed_dim, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_out, encoder_padding_mask, self_attn_mask, self_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x, x, x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask
        )
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(
            x, encoder_out, encoder_out, key_padding_mask=encoder_padding_mask
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, src_vocab):
        super().__init__()
        dim = cfg["encoder_embed_dim"]
        self.embed_scale = math.sqrt(dim)
        self.embed_tokens = nn.Embedding(src_vocab, dim, padding_idx=PAD_IDX)
        self.embed_positions = SinusoidalPositionalEmbedding(dim, PAD_IDX)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(dim, cfg["encoder_ffn_embed_dim"], cfg["encoder_attention_heads"])
            for _ in range(cfg["encoder_layers"])
        )
        self.layer_norm = nn.LayerNorm(dim)  # normalize_before, so a final LN
        self.register_buffer("version", torch.Tensor([3]))

    def forward(self, src_tokens):
        # src_tokens: [B, T]
        encoder_padding_mask = src_tokens.eq(PAD_IDX)  # [B, T]
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens)
        x = x.transpose(0, 1)  # [T, B, C]
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x, encoder_padding_mask


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, tgt_vocab):
        super().__init__()
        dim = cfg["decoder_embed_dim"]
        self.embed_scale = math.sqrt(dim)
        self.embed_tokens = nn.Embedding(tgt_vocab, dim, padding_idx=PAD_IDX)
        self.embed_positions = SinusoidalPositionalEmbedding(dim, PAD_IDX)
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(dim, cfg["decoder_ffn_embed_dim"], cfg["decoder_attention_heads"])
            for _ in range(cfg["decoder_layers"])
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, tgt_vocab, bias=False)
        self.register_buffer("version", torch.Tensor([3]))

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask):
        self_padding_mask = prev_output_tokens.eq(PAD_IDX)  # [B, T]
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + self.embed_positions(prev_output_tokens)
        x = x.transpose(0, 1)  # [T, B, C]

        seq_len = x.size(0)
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_padding_mask, causal, self_padding_mask)
        x = self.layer_norm(x)
        return x


class FairseqTransformer(nn.Module):
    def __init__(self, cfg, src_vocab, tgt_vocab):
        super().__init__()
        self.encoder = TransformerEncoder(cfg, src_vocab)
        self.decoder = TransformerDecoder(cfg, tgt_vocab)

    def forward(self, src_tokens, prev_output_tokens):
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        return self.decoder(prev_output_tokens, encoder_out, encoder_padding_mask)


def build_model_from_checkpoint(ckpt, src_vocab, tgt_vocab):
    """Instantiate the pure-torch model and load the fairseq state_dict strictly."""
    args = ckpt["args"]
    cfg = {
        k: getattr(args, k)
        for k in (
            "encoder_embed_dim", "encoder_ffn_embed_dim", "encoder_attention_heads",
            "encoder_layers", "decoder_embed_dim", "decoder_ffn_embed_dim",
            "decoder_attention_heads", "decoder_layers",
        )
    }
    model = FairseqTransformer(cfg, src_vocab, tgt_vocab)
    state = dict(ckpt["model"])
    # The position buffer is a runtime device/dtype holder, not a weight; our
    # module recreates it, so drop the checkpoint copy before the strict load.
    state.pop("encoder.embed_positions._float_tensor", None)
    state.pop("decoder.embed_positions._float_tensor", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the position buffers we deliberately recreate may be "missing"; any
    # other discrepancy means the architecture doesn't match the checkpoint.
    missing = [k for k in missing if "embed_positions._float_tensor" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch — missing={missing}, unexpected={unexpected}"
        )
    return model


# hooks + batching
class HookManager:
    """Collect per-layer outputs ([T, B, C]) from encoder/decoder layers."""

    def __init__(self):
        self._handles = []
        self.reps = {}

    def register(self, model):
        for i, layer in enumerate(model.encoder.layers):
            self._handles.append(layer.register_forward_hook(self._hook("encoder", i)))
        for i, layer in enumerate(model.decoder.layers):
            self._handles.append(layer.register_forward_hook(self._hook("decoder", i)))

    def _hook(self, layer_type, idx):
        key = (layer_type, idx)

        def fn(module, inp, out):
            self.reps.setdefault(key, []).append(out.detach().cpu())

        return fn

    def clear(self):
        self.reps = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def pad_rows(rows, pad_value=PAD_IDX):
    """Right-pad a list of variable-length id lists into a [B, T] LongTensor."""
    max_len = max(len(r) for r in rows)
    out = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def move_eos_to_beginning(rows):
    """fairseq prev_output_tokens: prepend eos, drop the trailing eos per row."""
    prev = []
    for r in rows:
        assert r[-1] == EOS_IDX, "target rows must end with eos before shifting"
        prev.append([EOS_IDX] + r[:-1])
    return prev


def content_mask_from_rows(rows, symbols, side):
    """[B, T] bool content mask aligned with the right-padded id tensor."""
    max_len = max(len(r) for r in rows)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, r in enumerate(rows):
        for j, tok_id in enumerate(r):
            mask[i, j] = is_content_symbol(symbols[tok_id], side)
    return mask


def pad_and_concatenate(tensor_list, pad_dim=1):
    """Pad along pad_dim to the global max, then concat along dim 0."""
    max_size = max(t.shape[pad_dim] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_amount = max_size - t.shape[pad_dim]
        if pad_amount > 0:
            shape = list(t.shape)
            shape[pad_dim] = pad_amount
            t = torch.cat([t, torch.zeros(shape, dtype=t.dtype)], dim=pad_dim)
        padded.append(t)
    return torch.cat(padded, dim=0)


# per-model extraction
def parse_split_run(model_dir):
    """``10L_90NL_1_1-models`` gives ('10L_90NL', '1_1')."""
    name = os.path.basename(model_dir.rstrip("/"))
    m = re.match(r"^(\d+L_\d+NL)_(\d+_\d+)-models$", name)
    if not m:
        raise ValueError(f"Cannot parse split/run from {name!r}")
    return m.group(1), m.group(2)


def resolve_checkpoint_file(model_dir):
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No checkpoint_best/last.pt in {model_dir}")


def extract_one(checkpoint, data_bin, run, output_path, test_src, test_tgt, batch_size):
    """Run one char_sep model forward over its test set and dump representations."""
    src_dict_path = os.path.join(data_bin, f"dict.{run}.src.txt")
    tgt_dict_path = os.path.join(data_bin, f"dict.{run}.tgt.txt")
    for p in (src_dict_path, tgt_dict_path, test_src, test_tgt):
        if not os.path.exists(p):
            print(f"Missing required file: {p}")
            return EXIT_ERROR

    src_symbols, src_sym2idx = load_fairseq_dict(src_dict_path)
    tgt_symbols, tgt_sym2idx = load_fairseq_dict(tgt_dict_path)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src_rows_dim = ckpt["model"]["encoder.embed_tokens.weight"].shape[0]
    tgt_rows_dim = ckpt["model"]["decoder.embed_tokens.weight"].shape[0]

    # Correctness gate: a dictionary that does not match the embedding table
    # would map characters to the wrong rows and yield meaningless vectors.
    if len(src_symbols) != src_rows_dim or len(tgt_symbols) != tgt_rows_dim:
        print(f"Dictionary/checkpoint vocab mismatch for run {run}: src dict={len(src_symbols)} vs embed={src_rows_dim}, tgt dict={len(tgt_symbols)} vs embed={tgt_rows_dim}. Point --data-bin at the matching seperate_char data directory.")
        return EXIT_ERROR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(ckpt, len(src_symbols), len(tgt_symbols))
    model.to(device).eval()

    with open(test_src, encoding="utf-8") as fh:
        src_lines = [ln.rstrip("\n") for ln in fh]
    with open(test_tgt, encoding="utf-8") as fh:
        tgt_lines = [ln.rstrip("\n") for ln in fh]
    if len(src_lines) != len(tgt_lines):
        print(f"test src/tgt length mismatch: {len(src_lines)} vs {len(tgt_lines)}")
        return EXIT_ERROR

    src_encoded = [encode_line(ln, src_sym2idx) for ln in src_lines]
    tgt_encoded = [encode_line(ln, tgt_sym2idx) for ln in tgt_lines]

    hooks = HookManager()
    hooks.register(model)

    batch_reps = {}
    batch_src_masks, batch_trg_masks = [], []
    batch_src_content, batch_trg_content = [], []

    with torch.no_grad():
        for start in range(0, len(src_encoded), batch_size):
            src_rows = src_encoded[start : start + batch_size]
            tgt_rows = tgt_encoded[start : start + batch_size]
            prev_rows = move_eos_to_beginning(tgt_rows)

            src_tokens = pad_rows(src_rows).to(device)
            prev_tokens = pad_rows(prev_rows).to(device)

            batch_src_masks.append(src_tokens.ne(PAD_IDX).cpu())
            batch_trg_masks.append(prev_tokens.ne(PAD_IDX).cpu())
            batch_src_content.append(content_mask_from_rows(src_rows, src_symbols, "encoder"))
            batch_trg_content.append(content_mask_from_rows(prev_rows, tgt_symbols, "decoder"))

            model(src_tokens, prev_tokens)

            for key, tensors in hooks.reps.items():
                # hook fired once per layer per batch, so tensors holds one [T,B,C]
                for t in tensors:
                    batch_reps.setdefault(key, []).append(t.transpose(0, 1))  # [B,T,C]
            hooks.clear()

    hooks.remove()

    if not batch_reps:
        print(f"No representations collected for run {run}")
        return EXIT_ERROR

    os.makedirs(output_path, exist_ok=True)
    src_masks = pad_and_concatenate(batch_src_masks, pad_dim=1)
    trg_masks = pad_and_concatenate(batch_trg_masks, pad_dim=1)
    src_content = pad_and_concatenate(batch_src_content, pad_dim=1)
    trg_content = pad_and_concatenate(batch_trg_content, pad_dim=1)

    embed_dim = None
    n_enc = n_dec = 0
    for (layer_type, idx), tensors in sorted(batch_reps.items()):
        rep = pad_and_concatenate(tensors, pad_dim=1)  # [n_samples, seq_len, C]
        embed_dim = rep.shape[2]
        torch.save(rep, os.path.join(output_path, f"{layer_type}_layer_{idx}.pt"))

        mask = src_masks if layer_type == "encoder" else trg_masks
        content = src_content if layer_type == "encoder" else trg_content
        torch.save(mask, os.path.join(output_path, f"{layer_type}_layer_{idx}_mask.pt"))
        torch.save(content, os.path.join(output_path, f"{layer_type}_layer_{idx}_content_mask.pt"))

        if layer_type == "encoder":
            n_enc += 1
        else:
            n_dec += 1

    metadata = {
        "embed_dim": embed_dim,
        "n_encoder_layers": n_enc,
        "n_decoder_layers": n_dec,
        "n_samples": int(src_masks.shape[0]),
        "src_seq_len": int(src_masks.shape[1]),
        "trg_seq_len": int(trg_masks.shape[1]),
        "framework": "fairseq",
        "arch": "transformer",
        "checkpoint": os.path.abspath(checkpoint),
        "src_vocab": len(src_symbols),
        "tgt_vocab": len(tgt_symbols),
        "mean_src_content_tokens": float(src_content.float().sum(dim=1).mean()),
        "mean_trg_content_tokens": float(trg_content.float().sum(dim=1).mean()),
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    return EXIT_SUCCESS


# CLI
def parse_args():
    sentinels = dict(standard_sentinels())
    sentinels["FEATURE_INFORMED_CHAR_SEP_CKPTS"] = os.path.join(
        FEATURE_INFORMED_ROOT, "checkpoints/char_sep/seperate_char_checkpoints")
    return parse(__doc__, types=dict(batch_size=int), sentinels=sentinels)


def default_test_files(data_bin, run, test_src, test_tgt):
    return (
        test_src or os.path.join(data_bin, f"test.{run}.src"),
        test_tgt or os.path.join(data_bin, f"test.{run}.tgt"),
    )


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        if not args.data_root:
            raise ValueError("--all requires --data-root")
        model_dirs = sorted(glob.glob(os.path.join(args.checkpoint_root, "*-models")))
        skipped = [d for d in model_dirs if d.endswith("_old") or d.endswith("_bk")]
        model_dirs = [d for d in model_dirs if not (d.endswith("_old") or d.endswith("_bk"))]
        for d in skipped:
            print(f"Skipping excluded model dir: {os.path.basename(d)}")

        any_error = False
        for model_dir in model_dirs:
            split, run = parse_split_run(model_dir)
            data_bin = os.path.join(args.data_root, f"{split}_{run}")
            checkpoint = resolve_checkpoint_file(model_dir)
            output_path = os.path.join(args.output_dir, args.model_type, f"{split}_{run}")
            if os.path.exists(os.path.join(output_path, "encoder_layer_0.pt")):
                print(f"Skipping {args.model_type}/{split}_{run} -- already extracted")
                continue
            test_src, test_tgt = default_test_files(data_bin, run, None, None)
            rc = extract_one(checkpoint, data_bin, run, output_path, test_src, test_tgt, args.batch_size)
            any_error = any_error or (rc == EXIT_ERROR)
        raise SystemExit(EXIT_ERROR if any_error else EXIT_SUCCESS)

    if not args.checkpoint or not args.data_bin:
        raise ValueError("Single-model mode requires --checkpoint and --data-bin (or use --all)")
    model_dir = os.path.dirname(args.checkpoint)
    split, run = parse_split_run(model_dir)
    output_path = os.path.join(args.output_dir, args.model_type, f"{split}_{run}")
    if os.path.exists(os.path.join(output_path, "encoder_layer_0.pt")):
        print(f"Skipping {args.model_type}/{split}_{run} -- already extracted")
        raise SystemExit(EXIT_SKIPPED)
    test_src, test_tgt = default_test_files(args.data_bin, run, args.test_src, args.test_tgt)
    rc = extract_one(args.checkpoint, args.data_bin, run, output_path, test_src, test_tgt, args.batch_size)
    raise SystemExit(rc)
