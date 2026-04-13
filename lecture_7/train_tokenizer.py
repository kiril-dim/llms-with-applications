#!/usr/bin/env python3
"""
train_tokenizer.py — Train a Bulgarian BPE tokenizer on a subset of chitanka.txt.

Trains a byte-level BPE tokenizer (same architecture as GPT-2) on Bulgarian text
and saves it for use in scaling_experiment.py.

Usage:
    python train_tokenizer.py --data_path ../../data/chitanka.txt
    python train_tokenizer.py --data_path ../../data/chitanka.txt --fraction 0.05
    python train_tokenizer.py --data_path ../../data/chitanka.txt --vocab_size 32000

Requirements:
    pip install tokenizers tiktoken
"""

import os
import json
import argparse


def iter_sampled_chunks(path: str, fraction: float, chunk_size: int = 512 * 1024):
    """
    Yield text chunks sampled evenly across the whole file.

    Divides the file into N equal segments and reads one chunk from each,
    so the tokenizer sees vocabulary from all parts of the corpus.
    """
    file_size = os.path.getsize(path)
    target_bytes = int(file_size * fraction)
    n_chunks = max(1, target_bytes // chunk_size)
    step = file_size // n_chunks

    with open(path, "rb") as f:
        for i in range(n_chunks):
            f.seek(i * step)
            f.readline()          # skip partial line at seek boundary
            raw = f.read(chunk_size)
            if raw:
                yield raw.decode("utf-8", errors="replace")


def train(data_path: str, output_dir: str, vocab_size: int, fraction: float):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.processors import ByteLevel as ByteLevelProcessor

    file_size_mb = os.path.getsize(data_path) / 1024 / 1024
    sample_mb = file_size_mb * fraction

    print(f"Corpus  : {data_path}  ({file_size_mb:.0f} MB)")
    print(f"Sample  : {fraction*100:.0f}%  (~{sample_mb:.0f} MB, evenly distributed)")
    print(f"Vocab   : {vocab_size:,}")
    print(f"Output  : {output_dir}/\n")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        min_frequency=2,
        show_progress=True,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        iter_sampled_chunks(data_path, fraction),
        trainer=trainer,
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"\nSaved to: {tokenizer_path}")

    # Save metadata so scaling_experiment.py can read vocab size
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "fraction": fraction,
        "source": os.path.basename(data_path),
    }
    with open(os.path.join(output_dir, "tokenizer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return tokenizer


def compare_fertility(tokenizer, data_path: str, sample_bytes: int = 500_000):
    """Compare tokens-per-character vs GPT-2 BPE on a small sample."""
    import tiktoken

    with open(data_path, encoding="utf-8", errors="replace") as f:
        sample = f.read(sample_bytes)

    gpt2 = tiktoken.get_encoding("gpt2")

    bg_tokens  = len(tokenizer.encode(sample).ids)
    gpt2_tokens = len(gpt2.encode_ordinary(sample))
    chars = len(sample)

    print("\n── Fertility comparison (tokens per character, lower is better) ──")
    print(f"  Sample        : {chars:,} characters")
    print(f"  Bulgarian BPE : {bg_tokens:,} tokens  ({bg_tokens/chars:.3f} tok/char)")
    print(f"  GPT-2 BPE     : {gpt2_tokens:,} tokens  ({gpt2_tokens/chars:.3f} tok/char)")
    print(f"  Reduction     : {(1 - bg_tokens/gpt2_tokens)*100:.1f}% fewer tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default="../../data/chitanka.txt")
    parser.add_argument("--output_dir", type=str, default="tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--fraction",   type=float, default=0.10,
                        help="Fraction of corpus to sample (default: 0.10 = 10%%)")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Corpus not found: {args.data_path}")

    tokenizer = train(args.data_path, args.output_dir, args.vocab_size, args.fraction)
    compare_fertility(tokenizer, args.data_path)


if __name__ == "__main__":
    main()
