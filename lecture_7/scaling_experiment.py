#!/usr/bin/env python3
"""
scaling_experiment.py — Pre-training script for L7 scaling law demonstration.

Trains 6 nanoGPT-style models of increasing size on Chitanka corpus,
then saves results to scaling_results.json for visualization in the lecture notebook.

Each model is trained on the same number of tokens so the comparison is fair
(equal data budget, not equal compute budget — simplest approach for a lecture demo).

Usage:
    # Use a local text file with custom Bulgarian tokenizer:
    python scaling_experiment.py --data_path /path/to/text.txt --tokenizer_path tokenizer/tokenizer.json

    # Fall back to GPT-2 BPE tokenizer (no --tokenizer_path):
    python scaling_experiment.py --data_path /path/to/text.txt

    # Train a subset of models (useful for testing):
    python scaling_experiment.py --models 1M,10M,124M

    # Shorter run for testing the setup:
    python scaling_experiment.py --training_tokens 2_000_000 --models 1M

Requirements:
    pip install torch tiktoken tokenizers datasets tqdm
"""

import contextlib
import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Model configurations
# Six sizes covering ~3 orders of magnitude in parameter count.
# Architecture follows nanoGPT / GPT-2 conventions.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 1024
    vocab_size: int = 50257   # GPT-2 BPE vocabulary
    bias: bool = False        # no bias (cleaner, like LLaMA)


MODEL_CONFIGS = [
    GPTConfig(name="1M",   n_layer=4,  n_head=4,  n_embd=128),
    GPTConfig(name="3M",   n_layer=4,  n_head=4,  n_embd=256),
    GPTConfig(name="10M",  n_layer=6,  n_head=6,  n_embd=384),
    GPTConfig(name="30M",  n_layer=6,  n_head=8,  n_embd=640),
    GPTConfig(name="124M", n_layer=12, n_head=12, n_embd=768),
    GPTConfig(name="355M", n_layer=24, n_head=16, n_embd=1024),
]

CONFIG_BY_NAME = {c.name: c for c in MODEL_CONFIGS}


# ──────────────────────────────────────────────────────────────────────────────
# Model architecture (self-contained nanoGPT implementation)
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        B, T, C = x.shape
        head_size = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        # uses Flash Attention kernel when available (PyTorch >= 2.0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act    = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scale residual projections by 1/sqrt(2 * n_layer) as in GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self) -> int:
        # exclude the tied embedding (counted once)
        return sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(
    data_path: Optional[str],
    cache_dir: str,
    tokenizer_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> tuple:
    """
    Tokenize data and return (train_ids, val_ids, vocab_size) as numpy uint16 arrays.

    Source priority:
      1. --data_path text file if provided
      2. Cached tokenized data (data_cache/train.bin, val.bin)
      3. Bulgarian Wikipedia via HuggingFace datasets (downloaded once, cached)

    Tokenizer priority:
      1. --tokenizer_path JSON file (custom Bulgarian BPE)
      2. GPT-2 BPE via tiktoken (fallback)

    max_tokens: if set, stop tokenizing after this many tokens (for smoke tests).

    90/10 train/val split.
    """
    if tokenizer_path and os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer as HFTokenizer
        _hf_tok = HFTokenizer.from_file(tokenizer_path)
        vocab_size = _hf_tok.get_vocab_size()
        print(f"Tokenizer: custom Bulgarian BPE  ({vocab_size:,} tokens)  [{tokenizer_path}]")
        def encode(text):
            return _hf_tok.encode(text).ids
    else:
        import tiktoken
        _gpt2 = tiktoken.get_encoding("gpt2")
        vocab_size = 50257
        print("Tokenizer: GPT-2 BPE (tiktoken fallback)")
        def encode(text):
            return _gpt2.encode_ordinary(text)

    cache_train = os.path.join(cache_dir, "train.bin")
    cache_val   = os.path.join(cache_dir, "val.bin")
    cache_info  = os.path.join(cache_dir, "info.json")

    if os.path.exists(cache_train) and os.path.exists(cache_val):
        print("Found cached tokenized data — loading...")
        train_ids = np.fromfile(cache_train, dtype=np.uint16)
        val_ids   = np.fromfile(cache_val,   dtype=np.uint16)
        if os.path.exists(cache_info):
            with open(cache_info) as f:
                info = json.load(f)
            print(f"  Source: {info.get('source', 'unknown')}")
        print(f"  Train: {len(train_ids):,} tokens  |  Val: {len(val_ids):,} tokens")
        return train_ids, val_ids, vocab_size

    os.makedirs(cache_dir, exist_ok=True)
    source_name = ""

    if data_path and os.path.exists(data_path):
        source_name = os.path.basename(data_path)
        cap = f"  (capped at {max_tokens/1e6:.1f}M tokens)" if max_tokens else ""
        print(f"Tokenizing: {data_path}{cap}")
        READ_CHUNK = 32 * 1024 * 1024   # 32 MB of text at a time
        all_ids = []
        bytes_read = 0
        file_size = os.path.getsize(data_path)
        with open(data_path, encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(READ_CHUNK)
                if not chunk:
                    break
                all_ids.extend(encode(chunk))
                bytes_read += len(chunk.encode("utf-8"))
                print(f"  {bytes_read/1e6:.0f}/{file_size/1e6:.0f} MB  —  "
                      f"{len(all_ids)/1e6:.1f}M tokens so far")
                if max_tokens and len(all_ids) >= max_tokens:
                    all_ids = all_ids[:max_tokens]
                    print(f"  Reached max_tokens cap ({max_tokens:,}) — stopping early.")
                    break
    else:
        print("Downloading Bulgarian Wikipedia (this takes a few minutes the first time)...")
        from datasets import load_dataset
        ds = load_dataset("wikipedia", "20231101.bg", split="train", trust_remote_code=True)
        texts = [row["text"] for row in ds]
        source_name = "Bulgarian Wikipedia (20231101)"
        print(f"  Loaded {len(texts):,} articles")
        print("Tokenizing...")
        all_ids = []
        report_every = max(1, len(texts) // 20)
        for i, text in enumerate(texts):
            all_ids.extend(encode(text))
            if (i + 1) % report_every == 0:
                print(f"  {i+1:,}/{len(texts):,} docs  —  {len(all_ids)/1e6:.1f}M tokens so far")
            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

    all_ids = np.array(all_ids, dtype=np.uint16)
    print(f"Total tokens: {len(all_ids):,}")

    split = int(0.9 * len(all_ids))
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    train_ids.tofile(cache_train)
    val_ids.tofile(cache_val)
    with open(cache_info, "w") as f:
        json.dump({"source": source_name, "total_tokens": len(all_ids)}, f)

    print(f"Saved to {cache_dir}/  (train: {len(train_ids):,}  val: {len(val_ids):,})")
    return train_ids, val_ids, vocab_size


class DataLoader:
    """Simple sequential data loader that wraps around at the end."""

    def __init__(self, data: np.ndarray, batch_size: int, block_size: int, device: str):
        self.data = torch.from_numpy(data.astype(np.int64))
        self.B = batch_size
        self.T = block_size
        self.device = device
        self.pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        if self.pos + B * T + 1 > len(self.data):
            self.pos = 0
        chunk = self.data[self.pos : self.pos + B * T + 1]
        x = chunk[:-1].view(B, T).to(self.device)
        y = chunk[1: ].view(B, T).to(self.device)
        self.pos += B * T
        return x, y

    def reset(self):
        self.pos = 0


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def batch_size_for(config: GPTConfig, device: str = "cuda") -> int:
    """
    Heuristic batch size. Tuned for a 24-32 GB CUDA GPU.
    On MPS or CPU, use a small fixed batch — grad_accum compensates.
    Gradient accumulation brings the effective batch to ~500K tokens per update.
    """
    if device != "cuda":
        return 2   # logits tensor = 2 × 1024 × vocab_size — safe on any device

    embd = config.n_embd
    if   embd <= 128:  return 64
    elif embd <= 256:  return 32
    elif embd <= 384:  return 16
    elif embd <= 640:  return 8
    elif embd <= 768:  return 4
    else:              return 2    # 355M / 1024-length sequences


def cosine_schedule(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_loss(model: GPT, loader: DataLoader, n_batches: int) -> float:
    model.eval()
    loader.reset()
    losses = [model(*(loader.next_batch()))[1].item() for _ in range(n_batches)]
    model.train()
    return float(np.mean(losses))


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_one_model(
    config: GPTConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    training_tokens: int,
    output_dir: str,
    device: str,
) -> dict:
    """
    Train one model for `training_tokens` tokens.
    Saves the best checkpoint (by val loss) and returns a results dict.
    """
    sep = "─" * 62
    print(f"\n{sep}")

    model = GPT(config).to(device)
    n_params = model.count_params()
    print(f"  Model : {config.name}  ({n_params:,} parameters)")
    print(f"  Arch  : {config.n_layer} layers × {config.n_head} heads × {config.n_embd} embd")

    # torch.compile speeds up training ~20-30% on CUDA; skip on CPU/MPS
    if device == "cuda" and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile ...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile failed ({e}), continuing without it.")

    B = batch_size_for(config, device)
    T = config.block_size

    # gradient accumulation to target ~500K tokens per optimizer step
    TARGET_TOKENS_PER_UPDATE = 524_288
    tokens_per_micro_step = B * T
    grad_accum = max(1, TARGET_TOKENS_PER_UPDATE // tokens_per_micro_step)
    tokens_per_update = tokens_per_micro_step * grad_accum

    total_steps  = training_tokens // tokens_per_update
    warmup_steps = max(100, total_steps // 20)
    eval_every   = max(50, total_steps // 100)   # ~100 eval points over training
    eval_batches = 50

    # learning rate: scale mildly with model size (larger models prefer lower lr)
    lr_max = 6e-4 * math.sqrt(128 / config.n_embd)   # anchored at 6e-4 for n_embd=128
    lr_max = max(lr_max, 1e-4)
    lr_min = lr_max / 10.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_max,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device == "cuda",
    )

    print(f"  Batch : {B} × seq {T}  |  grad_accum: {grad_accum}  "
          f"→  {tokens_per_update:,} tokens/update")
    print(f"  Steps : {total_steps:,}  (warmup: {warmup_steps:,})")
    print(f"  LR    : {lr_max:.2e} → {lr_min:.2e}")
    print(f"  Target: {training_tokens:,} tokens")
    print(sep)

    train_loader = DataLoader(train_data, B, T, device)
    val_loader   = DataLoader(val_data,   B, T, device)

    val_log   = []   # list of [step, tokens_seen, val_loss]
    train_log = []   # list of [step, tokens_seen, train_loss]
    best_val  = float("inf")
    ckpt_path = os.path.join(output_dir, f"ckpt_{config.name}.pt")
    csv_path  = os.path.join(output_dir, f"log_{config.name}.csv")
    t0 = time.time()
    tokens_seen = 0

    # Write CSV header (overwrites any previous run for this model)
    with open(csv_path, "w") as csv_f:
        csv_f.write("step,tokens_seen,train_loss,val_loss,tok_per_sec,elapsed_min\n")

    for step in range(total_steps + 1):

        # ── evaluation ──────────────────────────────────────────────────────
        if step % eval_every == 0 or step == total_steps:
            v_loss = eval_loss(model, val_loader,   eval_batches)
            t_loss = eval_loss(model, train_loader, eval_batches)
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"  step {step:6d}/{total_steps}  |  "
                  f"train {t_loss:.4f}  val {v_loss:.4f}  |  "
                  f"{tokens_seen/1e6:7.1f}M tok  |  "
                  f"{tok_per_sec/1e3:.1f}K tok/s  |  "
                  f"{elapsed/60:.1f} min")
            val_log.append([step, tokens_seen, v_loss])
            train_log.append([step, tokens_seen, t_loss])

            # Append to CSV immediately so it's readable mid-run
            with open(csv_path, "a") as csv_f:
                csv_f.write(f"{step},{tokens_seen},{t_loss:.6f},{v_loss:.6f},"
                            f"{tok_per_sec:.1f},{elapsed/60:.2f}\n")

            if v_loss < best_val:
                best_val = v_loss
                torch.save({
                    "config":   asdict(config),
                    "model":    model.state_dict(),
                    "step":     step,
                    "val_loss": v_loss,
                }, ckpt_path)

        if step == total_steps:
            break

        # ── lr schedule ─────────────────────────────────────────────────────
        lr = cosine_schedule(step, warmup_steps, total_steps, lr_max, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── forward + backward with gradient accumulation ───────────────────
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else contextlib.nullcontext()
        )
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            with autocast_ctx:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += tokens_per_update

    total_time = time.time() - t0
    print(f"\n  Finished in {total_time/60:.1f} min  |  Best val loss: {best_val:.4f}")
    print(f"  Checkpoint saved to: {ckpt_path}")

    return {
        "name":              config.name,
        "n_params":          n_params,
        "n_layer":           config.n_layer,
        "n_head":            config.n_head,
        "n_embd":            config.n_embd,
        "training_tokens":   tokens_seen,
        "best_val_loss":     best_val,
        "final_val_loss":    val_log[-1][2] if val_log else None,
        "final_train_loss":  train_log[-1][2] if train_log else None,
        "val_log":           val_log,    # [[step, tokens, loss], ...]
        "train_log":         train_log,
        "train_time_min":    round(total_time / 60, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train nanoGPT models at multiple scales for L7 scaling law demo."
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to a plain-text .txt file. If omitted, downloads Bulgarian Wikipedia.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="scaling_results",
        help="Directory for checkpoints and results JSON. (default: scaling_results/)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data_cache",
        help="Directory for cached tokenized data. (default: data_cache/)",
    )
    parser.add_argument(
        "--training_tokens", type=int, default=500_000_000,
        help="Tokens to train each model on. (default: 500M)  "
             "Use 10_000_000 for a quick smoke-test.",
    )
    parser.add_argument(
        "--models", type=str, default="1M,3M,10M,30M,124M,355M",
        help="Comma-separated model names to train. (default: all six)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to a custom HuggingFace tokenizer JSON. "
             "If omitted, falls back to GPT-2 BPE via tiktoken.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Stop tokenizing after this many tokens. Useful for smoke tests "
             "to avoid loading the full corpus. (default: tokenize everything)",
    )
    args = parser.parse_args()

    # ── device setup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
    elif device == "cpu":
        print("WARNING: no GPU found — training will be very slow on CPU.")

    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "scaling_results.json")

    # ── load any previously completed results (allows resuming) ─────────────
    all_results: dict = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f):
                all_results[r["name"]] = r
        print(f"\nResuming: found existing results for {list(all_results.keys())}")

    # ── prepare / load tokenized data ───────────────────────────────────────
    train_ids, val_ids, vocab_size = prepare_data(
        args.data_path, args.cache_dir, args.tokenizer_path, args.max_tokens
    )

    total_tokens = len(train_ids) + len(val_ids)
    print(f"\nDataset : {total_tokens/1e6:.1f}M tokens total  "
          f"(train: {len(train_ids)/1e6:.1f}M  val: {len(val_ids)/1e6:.1f}M)")
    print(f"Training tokens per model: {args.training_tokens/1e6:.0f}M  "
          f"({args.training_tokens / len(train_ids):.1f}x epochs on train set)")

    # ── train ────────────────────────────────────────────────────────────────
    requested = [n.strip() for n in args.models.split(",")]
    configs   = [CONFIG_BY_NAME[n] for n in requested if n in CONFIG_BY_NAME]
    # Apply vocab_size from the tokenizer to all configs
    for c in configs:
        c.vocab_size = vocab_size

    unknown = [n for n in requested if n not in CONFIG_BY_NAME]
    if unknown:
        print(f"WARNING: unknown model names ignored: {unknown}")
        print(f"  Valid names: {list(CONFIG_BY_NAME.keys())}")

    print(f"\nModels to train: {[c.name for c in configs]}\n")

    for config in configs:
        if config.name in all_results:
            print(f"Skipping {config.name} — already in {results_path}")
            continue

        result = train_one_model(
            config, train_ids, val_ids,
            args.training_tokens, args.output_dir, device,
        )
        all_results[config.name] = result

        # save after each model so a crash doesn't lose everything
        with open(results_path, "w") as f:
            json.dump(list(all_results.values()), f, indent=2)
        print(f"  Results saved to {results_path}")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n\n" + "═" * 62)
    print("  RESULTS SUMMARY")
    print("═" * 62)
    print(f"  {'Model':<8}  {'Params':>12}  {'Best Val Loss':>14}  {'Time (min)':>11}")
    print("  " + "─" * 56)
    for r in sorted(all_results.values(), key=lambda x: x["n_params"]):
        print(f"  {r['name']:<8}  {r['n_params']:>12,}  "
              f"{r['best_val_loss']:>14.4f}  {r['train_time_min']:>11.1f}")
    print("═" * 62)
    print(f"\nAll results saved to: {results_path}")
    print("Run the L7 notebook to visualize the scaling law.")


if __name__ == "__main__":
    main()
