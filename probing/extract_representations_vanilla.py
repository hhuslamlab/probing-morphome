"""Extract probing representations from the fairseq vanilla models.

The vanilla checkpoints are fairseq 0.10.2 ``transformer`` checkpoints (an
``args`` namespace plus a ``model`` state_dict); fairseq itself is deliberately
not a dependency (it pins Python 3.8), so the pre-norm transformer forward pass
is reconstructed in pure PyTorch from the state_dict.  Output layout is
identical to ``extract_representations.py``, so downstream scripts run unchanged.
With no arguments all vanilla models are processed; ``--model`` restricts to one.

Usage:
  extract_representations_vanilla.py [--checkpoints-dir DIR] [--checkpoint-name NAME]
                                     [--model NAME] [--data-dir DIR]
                                     [--fairseq-data-root DIR] [--output-dir DIR]
                                     [--batch-size N] [--num-threads N]
                                     [--validate-n N] [--min-token-acc X]
                                     [--overwrite] [--checkpoint FILE]
                                     [--src-dict FILE] [--tgt-dict FILE]
                                     [--test-src FILE] [--test-tgt FILE]
  extract_representations_vanilla.py (-h | --help)

Options:
  --checkpoints-dir DIR    Directory holding the <model>-models checkpoint
                           folders [default: FEATURE_INFORMED_VANILLA_CKPTS].
  --checkpoint-name NAME   Checkpoint file to load per model
                           [default: checkpoint_best.pt].
  --model NAME             Extract only this model (e.g. 50L_50NL_2_1);
                           default = all vanilla models.
  --data-dir DIR           Root of the raw train/test data
                           [default: FEATURE_INFORMED_DATA].
  --fairseq-data-root DIR  Optional root containing <split>/data-bin/<model>/
                           dict.<model>.{src,tgt}.txt; if absent, dicts are
                           reconstructed from the corpora under the data dir.
  --output-dir DIR         Output root [default: data/probing/representations].
  --batch-size N           Batch size for inference [default: 32].
  --num-threads N          torch intra-op CPU threads. Default 1: these
                           matmuls are tiny (seq~33, dim 256), so multiple
                           threads OVERSUBSCRIBE and slow the forward pass
                           4-5x. Run several models in parallel single-
                           threaded to use all cores [default: 1].
  --validate-n N           Greedy-decode this many test examples to compare
                           with predictions, 0 = off [default: 200].
  --min-token-acc X        Abort a model (don't write reps) if teacher-forced
                           token accuracy is below this [default: 0.5].
  --overwrite              Re-extract even if output already exists.
  --checkpoint FILE        Single-model override: explicit checkpoint .pt
                           (overrides checkpoint-name discovery; requires a
                           model name for output naming / split+run).
  --src-dict FILE          Single-model override: explicit source dict file.
  --tgt-dict FILE          Single-model override: explicit target dict file.
  --test-src FILE          Single-model override: explicit test .src file.
  --test-tgt FILE          Single-model override: explicit test .tgt file.
"""

import glob
import json
import math
import os
import re
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from probing import EXIT_SUCCESS, EXIT_ERROR, EXIT_SKIPPED, SPLITS, FEATURE_INFORMED_ROOT
from probing.utils.cli import parse, standard_sentinels
from probing.utils.hooks import HookManager
from probing.extract_representations import pad_and_concatenate

# fairseq Dictionary special symbols, in the order they are added (indices 0-3).
BOS, PAD, EOS, UNK = "<s>", "<pad>", "</s>", "<unk>"
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
_SPECIAL_SYMBOLS = (BOS, PAD, EOS, UNK)


# fairseq Dictionary (load from a dict file, or rebuild it from training data)
class FairseqDict:
    """Minimal re-implementation of fairseq's ``Dictionary`` (the four specials sit at indices 0 to 3)."""

    def __init__(self, symbols):
        # symbols: list excluding specials; specials are prepended here.
        self.symbols = list(_SPECIAL_SYMBOLS) + list(symbols)
        self.token2idx = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbols)

    def encode(self, line, append_eos=True):
        """Token ids for a whitespace-tokenised line (unknown maps to UNK)."""
        ids = [self.token2idx.get(tok, UNK_IDX) for tok in line.split()]
        if append_eos:
            ids.append(EOS_IDX)
        return ids

    def decode_tokens(self, ids):
        """Symbols for a list of ids (no special-token filtering)."""
        return [self.symbols[i] if 0 <= i < len(self.symbols) else UNK for i in ids]

    @classmethod
    def from_file(cls, path):
        symbols = []
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                # fairseq writes "<symbol> <count>"; the symbol may contain no
                # spaces here (IPA char or a single <TAG> token), so rsplit once.
                sym = line.rsplit(" ", 1)[0]
                symbols.append(sym)
        return cls(symbols)

    @classmethod
    def from_corpus(cls, corpus_path, padding_factor=8):
        """Rebuild a fairseq dictionary from a corpus, mirroring ``Dictionary.finalize`` ordering and padding."""
        from collections import Counter

        counter = Counter()
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                counter.update(line.split())
        symbols = [sym for sym, _ in counter.most_common()]

        total = len(_SPECIAL_SYMBOLS) + len(symbols)
        i = 0
        while total % padding_factor != 0:
            symbols.append(f"madeupword{i:04d}")
            total += 1
            i += 1
        return cls(symbols)


# Pure-PyTorch re-implementation of the fairseq pre-norm transformer
def _make_positions(tokens, padding_idx):
    """fairseq ``utils.make_positions``: position ids skipping padding (non-pad positions start at padding_idx+1)."""
    mask = tokens.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """fairseq non-learned sinusoidal positional embedding (table built lazily, not stored in the checkpoint)."""

    def __init__(self, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.zeros(1))
        self._weights = None  # lazily built sinusoid table

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, tokens):
        # tokens: [batch, seq_len]
        seq_len = tokens.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        if self._weights is None or self._weights.size(0) < max_pos:
            self._weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self._weights = self._weights.to(self._float_tensor.device)
        positions = _make_positions(tokens, self.padding_idx)
        return self._weights[positions]  # [batch, seq_len, dim]


class MultiheadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention, fairseq weight layout."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # query/key/value: [tgt_len, batch, embed_dim]
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        def shape(x, length):
            # reshape [length, batch, embed] to [batch*heads, length, head_dim]
            return (
                x.view(length, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        q = shape(q, tgt_len)
        k = shape(k, src_len)
        v = shape(v, src_len)

        attn = torch.bmm(q, k.transpose(1, 2))  # [batch*heads, tgt_len, src_len]

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]

        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, tgt_len, src_len)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [batch,1,1,src_len]
                float("-inf"),
            )
            attn = attn.view(bsz * self.num_heads, tgt_len, src_len)

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # [batch*heads, tgt_len, head_dim]
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """fairseq pre-norm encoder layer.  Returns the hidden state [T, B, C]."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, key_padding_mask=encoder_padding_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(F.relu(self.fc1(x)))
        x = residual + x
        return x


class TransformerDecoderLayer(nn.Module):
    """fairseq pre-norm decoder layer.  Returns the hidden state [T, B, C]."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = MultiheadAttention(dim, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_out, encoder_padding_mask, self_attn_mask, self_attn_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x, x, x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(
            x, encoder_out, encoder_out,
            key_padding_mask=encoder_padding_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(F.relu(self.fc1(x)))
        x = residual + x
        return x


class FairseqEncoder(nn.Module):
    def __init__(self, vocab_size, dim, ffn_dim, heads, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
        self.embed_positions = SinusoidalPositionalEmbedding(dim, padding_idx=PAD_IDX)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, ffn_dim, heads) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim)  # pre-norm, so a final LN is present
        self.embed_scale = math.sqrt(dim)
        self.register_buffer("version", torch.tensor([3.0]))

    def forward(self, src_tokens):
        # src_tokens: [batch, src_len]
        padding_mask = src_tokens.eq(PAD_IDX)  # [batch, src_len], True = pad
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens)
        x = x.transpose(0, 1)  # [src_len, batch, dim]
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.layer_norm(x)
        return x, padding_mask  # encoder_out [src_len, batch, dim]


class FairseqDecoder(nn.Module):
    def __init__(self, vocab_size, dim, ffn_dim, heads, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
        self.embed_positions = SinusoidalPositionalEmbedding(dim, padding_idx=PAD_IDX)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dim, ffn_dim, heads) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)  # tied weight
        self.embed_scale = math.sqrt(dim)
        self.register_buffer("version", torch.tensor([3.0]))

    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask):
        # prev_output_tokens: [batch, tgt_len]
        self_attn_padding_mask = prev_output_tokens.eq(PAD_IDX)  # [batch, tgt_len]
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + self.embed_positions(prev_output_tokens)
        x = x.transpose(0, 1)  # [tgt_len, batch, dim]

        tgt_len = x.size(0)
        causal = torch.triu(
            x.new_full((tgt_len, tgt_len), float("-inf")), diagonal=1
        )

        for layer in self.layers:
            x = layer(
                x, encoder_out, encoder_padding_mask,
                self_attn_mask=causal,
                self_attn_padding_mask=self_attn_padding_mask,
            )
        x = self.layer_norm(x)
        logits = self.output_projection(x)  # [tgt_len, batch, vocab]
        return logits


class FairseqTransformer(nn.Module):
    """Pre-norm encoder-decoder transformer matching the fairseq state_dict."""

    def __init__(self, src_vocab, tgt_vocab, dim, ffn_dim, heads, n_enc, n_dec):
        super().__init__()
        self.encoder = FairseqEncoder(src_vocab, dim, ffn_dim, heads, n_enc)
        self.decoder = FairseqDecoder(tgt_vocab, dim, ffn_dim, heads, n_dec)

    def forward(self, src_tokens, prev_output_tokens):
        encoder_out, enc_padding_mask = self.encoder(src_tokens)
        logits = self.decoder(prev_output_tokens, encoder_out, enc_padding_mask)
        return logits


def build_model_from_checkpoint(ckpt, src_vocab, tgt_vocab, device):
    """Instantiate the transformer from the fairseq ``args`` and load weights."""
    args = ckpt["args"]
    model = FairseqTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        dim=args.encoder_embed_dim,
        ffn_dim=args.encoder_ffn_embed_dim,
        heads=args.encoder_attention_heads,
        n_enc=args.encoder_layers,
        n_dec=args.decoder_layers,
    )

    state = dict(ckpt["model"])
    # share_decoder_input_output_embed=True: the checkpoint may omit
    # output_projection.weight (tied to embed_tokens). Supply it if absent.
    if "decoder.output_projection.weight" not in state:
        state["decoder.output_projection.weight"] = state["decoder.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The only acceptable mismatches are the lazily-built sinusoid tables, which
    # are not parameters. Everything else must line up exactly.
    real_missing = [k for k in missing if not k.endswith("embed_positions._weights")]
    if real_missing:
        raise RuntimeError(f"Missing checkpoint keys: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


# Data handling
def split_of(model_name):
    """Split prefix of a model name: '50L_50NL_2_1' gives '50L_50NL'."""
    return re.sub(r"_\d+_\d+$", "", model_name)


def run_of(model_name):
    """Run suffix of a model name: '50L_50NL_2_1' gives '2_1'."""
    m = re.search(r"_(\d+_\d+)$", model_name)
    return m.group(1) if m else None


def resolve_dict(lang, model_name, split, args, explicit_path=None):
    """Return (FairseqDict, source_path), from a dict file if present else rebuilt from the training corpus."""
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Explicit {lang} dict not found: {explicit_path}")
        return FairseqDict.from_file(explicit_path), explicit_path

    # 1) fairseq data-bin dict file, if a data root was given.
    dict_path = None
    if args.fairseq_data_root:
        dict_path = os.path.join(
            args.fairseq_data_root, split, "data-bin", model_name,
            f"dict.{model_name}.{lang}.txt",
        )
        if os.path.exists(dict_path):
            return FairseqDict.from_file(dict_path), dict_path

    # 2) Reconstruct from the training corpus for that language.
    run = run_of(model_name)
    run_num = run.split("_")[0]
    corpus = os.path.join(
        args.data_dir, split, "train", f"run{run_num}",
        f"train.{model_name}.{lang}",
    )
    if not os.path.exists(corpus):
        tried = f"tried {dict_path} and {corpus}" if dict_path else f"tried {corpus}"
        raise FileNotFoundError(
            f"No {lang} dict file and no training corpus to rebuild it: {tried}"
        )
    return FairseqDict.from_corpus(corpus), corpus


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def make_batches(src_lines, tgt_lines, src_dict, tgt_dict, batch_size):
    """Yield padded batches in corpus order (prev_output_tokens uses fairseq's move-eos-to-beginning)."""
    for start in range(0, len(src_lines), batch_size):
        s_strs = src_lines[start:start + batch_size]
        t_strs = tgt_lines[start:start + batch_size]

        src_ids = [src_dict.encode(s, append_eos=True) for s in s_strs]
        tgt_ids = [tgt_dict.encode(t, append_eos=True) for t in t_strs]

        src_max = max(len(x) for x in src_ids)
        tgt_max = max(len(x) for x in tgt_ids)

        src = torch.full((len(src_ids), src_max), PAD_IDX, dtype=torch.long)
        target = torch.full((len(tgt_ids), tgt_max), PAD_IDX, dtype=torch.long)
        prev = torch.full((len(tgt_ids), tgt_max), PAD_IDX, dtype=torch.long)
        for i, ids in enumerate(src_ids):
            src[i, : len(ids)] = torch.tensor(ids)
        for i, ids in enumerate(tgt_ids):
            target[i, : len(ids)] = torch.tensor(ids)
            prev[i, 0] = EOS_IDX
            prev[i, 1 : len(ids)] = torch.tensor(ids[:-1])

        yield src, prev, target, s_strs, t_strs


def _is_tag(tok):
    return tok.startswith("<") and tok.endswith(">")


def content_mask_from_strings(strings, side, seq_len):
    """Bool mask [n, seq_len], True at character positions (decoder positions shifted +1 for the leading EOS)."""
    mask = torch.zeros(len(strings), seq_len, dtype=torch.bool)
    for i, s in enumerate(strings):
        toks = s.split()
        offset = 1 if side == "decoder" else 0  # decoder input is shifted by the leading EOS
        for j, tok in enumerate(toks):
            pos = j + offset
            if pos >= seq_len:
                break
            if side == "encoder" and (_is_tag(tok) or tok == "#"):
                continue
            mask[i, pos] = True
    return mask


def tag_mask_from_strings(strings, seq_len):
    """Bool mask [n, seq_len], True at encoder ``<TAG>`` positions only."""
    mask = torch.zeros(len(strings), seq_len, dtype=torch.bool)
    for i, s in enumerate(strings):
        for j, tok in enumerate(s.split()):
            if j >= seq_len:
                break
            if _is_tag(tok):
                mask[i, j] = True
    return mask


# Validation against saved predictions
def teacher_forced_token_accuracy(logits, target):
    """Return (correct, total) token counts of teacher-forced argmax over non-pad target positions."""
    pred = logits.argmax(-1).transpose(0, 1)  # [batch, tgt_len]
    valid = target.ne(PAD_IDX)
    correct = ((pred == target) & valid).sum().item()
    total = valid.sum().item()
    return correct, total


@torch.no_grad()
def greedy_decode(model, src_dict, tgt_dict, src_lines, max_len, device):
    """Autoregressive greedy decode; returns joined character strings for validation."""
    preds = []
    for line in src_lines:
        src = torch.tensor([src_dict.encode(line, append_eos=True)], device=device)
        encoder_out, enc_mask = model.encoder(src)
        ys = torch.tensor([[EOS_IDX]], device=device)  # fairseq starts with EOS
        out_syms = []
        for _ in range(max_len):
            logits = model.decoder(ys, encoder_out, enc_mask)
            nxt = int(logits[-1, 0].argmax())
            if nxt == EOS_IDX:
                break
            out_syms.append(tgt_dict.symbols[nxt])
            ys = torch.cat([ys, torch.tensor([[nxt]], device=device)], dim=1)
        preds.append("".join(out_syms))
    return preds


# Per-model extraction
def extract_one(model_dir, model_name, args, device):
    split = split_of(model_name)
    run = run_of(model_name)
    run_num = run.split("_")[0]

    output_path = os.path.join(args.output_dir, "vanilla", f"{split}_{run}")
    if os.path.exists(os.path.join(output_path, "encoder_layer_0.pt")) and not args.overwrite:
        print(f"Skipping {model_name} -- already extracted")
        return EXIT_SKIPPED

    # Path overrides (single-model mode) let us point at a checkpoint/dict/test
    # set that lives outside the standard layout (e.g. the naacl25 10L_90NL_1_1).
    checkpoint = args.checkpoint or os.path.join(model_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        return EXIT_ERROR

    test_src = args.test_src or os.path.join(args.data_dir, split, "test", f"run{run_num}", f"test.{model_name}.src")
    test_tgt = args.test_tgt or os.path.join(args.data_dir, split, "test", f"run{run_num}", f"test.{model_name}.tgt")
    for p in (test_src, test_tgt):
        if not os.path.exists(p):
            print(f"Missing test file: {p}")
            return EXIT_ERROR

    src_dict, src_dict_src = resolve_dict("src", model_name, split, args, explicit_path=args.src_dict)
    tgt_dict, tgt_dict_src = resolve_dict("tgt", model_name, split, args, explicit_path=args.tgt_dict)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Sanity: embedding rows must match the resolved vocab sizes, or the dict is wrong.
    enc_rows = ckpt["model"]["encoder.embed_tokens.weight"].shape[0]
    dec_rows = ckpt["model"]["decoder.embed_tokens.weight"].shape[0]
    if enc_rows != len(src_dict) or dec_rows != len(tgt_dict):
        print(f"Vocab size mismatch for {model_name}: checkpoint enc/dec={enc_rows}/{dec_rows} vs dict {len(src_dict)}/{len(tgt_dict)} (src dict from {src_dict_src}, tgt dict from {tgt_dict_src}). Refusing to extract.")
        return EXIT_ERROR

    model = build_model_from_checkpoint(ckpt, len(src_dict), len(tgt_dict), device)

    src_lines = read_lines(test_src)
    tgt_lines = read_lines(test_tgt)
    if len(src_lines) != len(tgt_lines):
        print(f"src/tgt line count mismatch: {len(src_lines)} vs {len(tgt_lines)}")
        return EXIT_ERROR

    hook_manager = HookManager()
    hook_manager.register_hooks(model)

    batch_reps = {}
    batch_src_masks, batch_trg_masks = [], []
    batch_src_content, batch_trg_content = [], []
    batch_src_tag = []
    tf_correct = tf_total = 0

    n_batches = (len(src_lines) + args.batch_size - 1) // args.batch_size
    log_every = max(1, n_batches // 20)  # ~20 progress lines per model
    with torch.no_grad():
        for bi, (src, prev, target, s_strs, t_strs) in enumerate(make_batches(
            src_lines, tgt_lines, src_dict, tgt_dict, args.batch_size
        )):
            if bi % log_every == 0:
                pass
            src = src.to(device)
            prev = prev.to(device)
            target_dev = target.to(device)

            logits = model(src, prev)  # registers hook outputs
            c, t = teacher_forced_token_accuracy(logits, target_dev)
            tf_correct += c
            tf_total += t

            # Padding masks (True = valid), oriented [batch, seq_len].
            batch_src_masks.append(src.ne(PAD_IDX).cpu())
            batch_trg_masks.append(prev.ne(PAD_IDX).cpu())
            batch_src_content.append(content_mask_from_strings(s_strs, "encoder", src.shape[1]))
            batch_trg_content.append(content_mask_from_strings(t_strs, "decoder", prev.shape[1]))
            batch_src_tag.append(tag_mask_from_strings(s_strs, src.shape[1]))

            reps = hook_manager.get_representations()
            for key, tensors in reps.items():
                batch_reps.setdefault(key, [])
                for tensor in tensors:
                    batch_reps[key].append(tensor.transpose(0, 1))  # [T,B,C] to [B,T,C]
            hook_manager.clear()

    hook_manager.remove_hooks()

    if not batch_reps:
        print(f"No representations collected for {model_name}")
        return EXIT_ERROR

    tf_acc = tf_correct / max(tf_total, 1)

    # Greedy exact-match validation against the GOLD test targets, on a subset.
    # (We compare to the gold target, not predictions_vanilla/*.txt, because those
    # files are post-processed -- e.g. the primary-stress mark 'ˈ' is stripped --
    # so they are not a faithful reference for the model's raw output.)
    greedy_match = None
    if args.validate_n > 0:
        n_val = min(args.validate_n, len(src_lines))
        max_len = max(len(t.split()) for t in tgt_lines) + 5
        decoded = greedy_decode(model, src_dict, tgt_dict, src_lines[:n_val], max_len, device)
        gold = ["".join(t.split()) for t in tgt_lines[:n_val]]
        match = sum(1 for a, b in zip(decoded, gold) if a == b)
        greedy_match = match / n_val

    if tf_acc < args.min_token_acc:
        print(f"  teacher-forced token accuracy {tf_acc:.4f} below threshold {args.min_token_acc:.4f} for {model_name} -- likely a wrong/mismatched dictionary. NOT writing representations.")
        return EXIT_ERROR

    # Pad to global max seq_len and concatenate, then save (mirrors extract_representations.py).
    os.makedirs(output_path, exist_ok=True)
    src_masks = pad_and_concatenate(batch_src_masks, pad_dim=1)
    trg_masks = pad_and_concatenate(batch_trg_masks, pad_dim=1)
    src_content = pad_and_concatenate(batch_src_content, pad_dim=1)
    trg_content = pad_and_concatenate(batch_trg_content, pad_dim=1)
    src_tag = pad_and_concatenate(batch_src_tag, pad_dim=1)

    n_samples = src_masks.shape[0]
    embed_dim = None
    n_enc = n_dec = 0
    for (layer_type, layer_index), tensors in sorted(batch_reps.items()):
        rep = pad_and_concatenate(tensors, pad_dim=1)  # [n, seq, dim]
        embed_dim = rep.shape[2]
        torch.save(rep, os.path.join(output_path, f"{layer_type}_layer_{layer_index}.pt"))

        mask = src_masks if layer_type == "encoder" else trg_masks
        torch.save(mask, os.path.join(output_path, f"{layer_type}_layer_{layer_index}_mask.pt"))
        content = src_content if layer_type == "encoder" else trg_content
        torch.save(content, os.path.join(output_path, f"{layer_type}_layer_{layer_index}_content_mask.pt"))

        # Tag-position masks exist on the encoder side only (target has no tags).
        if layer_type == "encoder":
            torch.save(src_tag, os.path.join(output_path, f"{layer_type}_layer_{layer_index}_tag_mask.pt"))

        if layer_type == "encoder":
            n_enc += 1
        else:
            n_dec += 1

    metadata = {
        "framework": "fairseq",
        "arch": ckpt["args"].arch,
        "checkpoint": checkpoint,
        "src_dict_source": src_dict_src,
        "tgt_dict_source": tgt_dict_src,
        "src_vocab_size": len(src_dict),
        "tgt_vocab_size": len(tgt_dict),
        "embed_dim": embed_dim,
        "n_encoder_layers": n_enc,
        "n_decoder_layers": n_dec,
        "n_samples": n_samples,
        "src_seq_len": int(src_masks.shape[1]),
        "trg_seq_len": int(trg_masks.shape[1]),
        "teacher_forced_token_acc": tf_acc,
        "greedy_exact_match": greedy_match,
        "mean_src_content_tokens": float(src_content.float().sum(dim=1).mean()),
        "mean_trg_content_tokens": float(trg_content.float().sum(dim=1).mean()),
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return EXIT_SUCCESS


def discover_models(checkpoints_dir):
    """Vanilla '<split>_<run>-models' dirs only; skip *_old / *_bk / runless."""
    keep = []
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, "*-models"))):
        name = os.path.basename(path)
        if name.endswith("-models_old") or name.endswith("-models_bk"):
            continue
        model_name = name[: -len("-models")]
        if split_of(model_name) not in SPLITS or run_of(model_name) is None:
            print(f"Skipping {name} (no <split>_<run> pattern)")
            continue
        keep.append((path, model_name))
    return keep


def parse_args():
    sentinels = dict(standard_sentinels())
    sentinels["FEATURE_INFORMED_VANILLA_CKPTS"] = os.path.join(
        FEATURE_INFORMED_ROOT, "checkpoints", "vanilla", "fixed_checkpoints")
    return parse(__doc__,
                 types=dict(batch_size=int, num_threads=int, validate_n=int,
                            min_token_acc=float),
                 sentinels=sentinels)


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit single-model mode: a checkpoint path is given for a model that
    # need not live under --checkpoints-dir (e.g. the naacl25 10L_90NL_1_1).
    if args.checkpoint:
        if not args.model:
            raise ValueError("--checkpoint requires --model (used for output naming / split+run)")
        if split_of(args.model) not in SPLITS or run_of(args.model) is None:
            raise ValueError(f"--model {args.model} must look like <split>_<run> (e.g. 10L_90NL_1_1)")
        models = [(None, args.model)]
    else:
        models = discover_models(args.checkpoints_dir)
        if args.model:
            models = [(p, m) for p, m in models if m == args.model]
            if not models:
                raise FileNotFoundError(f"Model {args.model} not found among vanilla -models dirs")

    results = {EXIT_SUCCESS: 0, EXIT_SKIPPED: 0, EXIT_ERROR: 0}
    failed = []
    for model_dir, model_name in models:
        try:
            code = extract_one(model_dir, model_name, args, device)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"Extraction failed for {model_name}")
            code = EXIT_ERROR
        results[code] = results.get(code, 0) + 1
        if code == EXIT_ERROR:
            failed.append(model_name)

    raise SystemExit(EXIT_ERROR if failed else EXIT_SUCCESS)
