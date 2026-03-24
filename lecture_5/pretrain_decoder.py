"""
Pretrain a small GPT (decoder-only) model on the Ivan Vazov corpus.

Usage:
    uv run python lectures/lecture_05/pretrain_decoder.py
    uv run python lectures/lecture_05/pretrain_decoder.py --epochs 5 --n_layer 6 --n_head 6 --n_embd 384

Outputs (in lectures/lecture_05/checkpoints/):
    decoder_vazov.pt   - model checkpoint (weights + config + tokenizer path)
    decoder_loss.json  - training loss history
    tokenizer.json     - BPE tokenizer trained on the corpus
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# ---- Model ----


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # Use PyTorch scaled_dot_product_attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"])

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["block_size"] :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---- Tokenizer ----


def train_tokenizer(corpus_path, vocab_size, save_path):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>"],
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


# ---- Data ----


def load_data(corpus_path, tokenizer, block_size, val_fraction=0.1):
    text = Path(corpus_path).read_text(encoding="utf-8")
    encoded = tokenizer.encode(text)
    tokens = torch.tensor(encoded.ids, dtype=torch.long)
    n = len(tokens)
    split = int(n * (1 - val_fraction))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    print(f"Corpus: {n} tokens, train: {len(train_tokens)}, val: {len(val_tokens)}")
    return train_tokens, val_tokens


def get_batch(tokens, block_size, batch_size, device):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---- Training ----


@torch.no_grad()
def estimate_loss(
    model, train_tokens, val_tokens, block_size, batch_size, device, eval_iters=50
):
    model.eval()
    out = {}
    for name, tokens in [("train", train_tokens), ("val", val_tokens)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(tokens, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(args):
    # Paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("vazov-gpt")
    run = mlflow.start_run()
    mlflow.set_tags(
        {
            "model_type": "nanoGPT",
            "corpus": "ivan_vazov",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )

    corpus_path = args.corpus
    tokenizer_path = str(out_dir / "tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    if os.path.exists(tokenizer_path) and not args.retrain_tokenizer:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = train_tokenizer(corpus_path, args.vocab_size, tokenizer_path)

    actual_vocab_size = tokenizer.get_vocab_size()

    # Config
    config = {
        "vocab_size": actual_vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
    }
    print(f"Model config: {config}")

    # Data
    train_tokens, val_tokens = load_data(corpus_path, tokenizer, args.block_size)

    # Model
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    mlflow.log_param("n_params", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Learning rate schedule: warmup + cosine decay to min_lr
    total_steps = args.epochs * (
        len(train_tokens) // (args.block_size * args.batch_size)
    )
    warmup_steps = (
        args.warmup_steps if args.warmup_steps > 0 else max(10, total_steps // 20)
    )
    print(
        f"Warmup steps: {warmup_steps} / {total_steps} total ({100 * warmup_steps / total_steps:.1f}%)"
    )

    mlflow.log_params(
        {
            **config,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
    )

    def get_lr(step):
        if step < warmup_steps:
            return args.min_lr + (args.lr - args.min_lr) * step / max(1, warmup_steps)
        decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # Training loop
    loss_history = {
        "train": [],
        "val": [],
        "step": [],
        "step_loss": [],
        "step_loss_step": [],
    }
    steps_per_epoch = len(train_tokens) // (args.block_size * args.batch_size)
    global_step = 0

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for step in range(steps_per_epoch):
            # Update learning rate
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = get_batch(train_tokens, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()

            step_loss = loss.item()
            loss_history["step_loss"].append(step_loss)
            loss_history["step_loss_step"].append(global_step)
            epoch_loss += step_loss
            n_batches += 1
            global_step += 1

            mlflow.log_metric("grad_norm", grad_norm, step=global_step)

            if (
                global_step % args.log_interval == 0
                and global_step % args.eval_interval != 0
            ):
                print(f"  step {global_step:5d} | loss {step_loss:.4f} | lr {lr:.2e}")

            # Eval every eval_interval steps
            if global_step % args.eval_interval == 0:
                losses = estimate_loss(
                    model,
                    train_tokens,
                    val_tokens,
                    args.block_size,
                    args.batch_size,
                    device,
                )
                loss_history["train"].append(losses["train"])
                loss_history["val"].append(losses["val"])
                loss_history["step"].append(global_step)
                print(
                    f"  step {global_step:5d} | "
                    f"train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | "
                    f"lr {lr:.2e}"
                )

                train_ppl = math.exp(losses["train"])
                val_ppl = math.exp(losses["val"])
                mlflow.log_metric("train_loss", losses["train"], step=global_step)
                mlflow.log_metric("val_loss", losses["val"], step=global_step)
                mlflow.log_metric("train_perplexity", train_ppl, step=global_step)
                mlflow.log_metric("val_perplexity", val_ppl, step=global_step)
                mlflow.log_metric(
                    "val_train_gap", losses["val"] - losses["train"], step=global_step
                )
                mlflow.log_metric("lr", lr, step=global_step)

        dt = time.time() - t0
        avg_loss = epoch_loss / n_batches
        tokens_per_sec = (steps_per_epoch * args.batch_size * args.block_size) / dt
        mlflow.log_metric("tokens_per_sec", tokens_per_sec, step=epoch + 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} done in {dt:.1f}s | avg train loss {avg_loss:.4f} | {tokens_per_sec:.0f} tok/s"
        )
        snapshot_epochs = {
            1,
            args.epochs // 4,
            args.epochs // 2,
            args.epochs * 3 // 4,
            args.epochs,
        }
        if (epoch + 1) in snapshot_epochs:
            snap_path = out_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                },
                snap_path,
            )
            # Generate a sample and log to MLflow
            model.eval()
            prompt = "Вазов"
            enc = tokenizer.encode(prompt)
            idx = torch.tensor([enc.ids], dtype=torch.long, device=device)
            gen = model.generate(idx, max_new_tokens=150)
            sample = tokenizer.decode(gen[0].tolist())
            mlflow.log_text(sample, f"samples/epoch_{epoch + 1}.txt")
            print(f"  → snapshot saved: {snap_path.name}")
            model.train()

    # Final eval
    final_losses = estimate_loss(
        model,
        train_tokens,
        val_tokens,
        args.block_size,
        args.batch_size,
        device,
    )
    print(
        f"\nFinal: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}"
    )

    # Save checkpoint
    ckpt_path = out_dir / "decoder_vazov.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "tokenizer_path": tokenizer_path,
            "final_train_loss": final_losses["train"],
            "final_val_loss": final_losses["val"],
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")

    mlflow.end_run()

    # Save loss history
    loss_path = out_dir / "decoder_loss.json"
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved to {loss_path}")

    # Generate sample text
    print("\n--- Sample generation ---")
    model.eval()
    prompt = "Вазов"
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=200)
    text = tokenizer.decode(generated[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated:\n{text}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain nanoGPT on Ivan Vazov corpus"
    )

    # Data
    parser.add_argument("--corpus", type=str, default="data/ivan_vazov.txt")
    parser.add_argument("--out_dir", type=str, default="lecture_5/checkpoints")

    # Tokenizer
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--retrain_tokenizer", action="store_true")

    # Model
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--block_size", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=3e-5,
        help="Minimum LR at end of cosine decay (default: 10%% of lr).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps. 0 = auto (5%% of total steps).",
    )
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
