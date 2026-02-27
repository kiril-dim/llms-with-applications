"""Помощни функции за предобработка на текстов корпус и генерация на текст."""

from collections import Counter

import torch
import torch.nn.functional as F


def load_corpus(path: str) -> str:
    """Зарежда текстов корпус от файл."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def preprocess(text: str, lowercase: bool = True, top_n: int = 40) -> str:
    """Привежда текста в малки букви и запазва само top_n най-чести символа."""
    if lowercase:
        text = text.lower()
    counts = Counter(text)
    kept = {c for c, _ in counts.most_common(top_n)}
    return "".join(c for c in text if c in kept)


def build_vocab(text: str) -> tuple[list, dict, dict]:
    """Строи речник: list от символи + двупосочно съответствие индекс ↔ символ."""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


def train_test_split(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Разделя текста на тренировъчен и тест набор."""
    split_idx = int(len(text) * train_ratio)
    return text[:split_idx], text[split_idx:]


def build_context_dataset(
    text: str, context_size: int, char_to_idx: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Строи dataset с контекстен прозорец от context_size символа."""
    xs, ys = [], []
    for i in range(context_size, len(text)):
        context = [char_to_idx[text[i - context_size + j]] for j in range(context_size)]
        xs.append(context)
        ys.append(char_to_idx[text[i]])
    return torch.tensor(xs), torch.tensor(ys)


def evaluate_linear(W, b, X, Y, vocab_size, batch_size=4096):
    """Оценява линеен модел: one-hot вход → logits = x @ W + b."""
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        for i in range(0, len(X), batch_size):
            xb = F.one_hot(X[i:i+batch_size], num_classes=vocab_size).float()
            yb = Y[i:i+batch_size]
            logits = xb @ W + b
            total_loss += F.cross_entropy(logits, yb, reduction='sum').item()
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def evaluate_context_linear(W, b, X, Y, vocab_size, batch_size=4096):
    """Оценява линеен модел с контекстен вход (encode_context)."""
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        for i in range(0, len(X), batch_size):
            xb = encode_context(X[i:i+batch_size], vocab_size)
            yb = Y[i:i+batch_size]
            logits = xb @ W + b
            total_loss += F.cross_entropy(logits, yb, reduction='sum').item()
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def evaluate_nn(model, X, Y, vocab_size, one_hot=False, batch_size=4096):
    """Оценява nn.Module модел (с или без one-hot кодиране на входа)."""
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            if one_hot:
                xb = F.one_hot(xb, num_classes=vocab_size).float()
            yb = Y[i:i+batch_size]
            logits = model(xb)
            total_loss += F.cross_entropy(logits, yb, reduction='sum').item()
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def encode_context(X, vocab_size):
    """Кодира контекстен вход като конкатенирани one-hot вектори."""
    B, C = X.shape
    return F.one_hot(X, num_classes=vocab_size).float().view(B, C * vocab_size)


def generate(
    model,
    char_to_idx: dict,
    idx_to_char: dict,
    length: int = 300,
    seed_char: str = "\n",
    one_hot: bool = False,
) -> str:
    """Генерира текст от даден модел.

    model     — извикуем обект (nn.Module или lambda), приема тензор и връща логити.
    one_hot   — ако True, входът се кодира като one-hot вектор (за линеен модел и CharNN);
                ако False, подава се директно индексът (за Embedding модели).
    """
    vocab_size = len(char_to_idx)
    result = [seed_char]
    idx = char_to_idx[seed_char]
    for _ in range(length):
        x = (
            F.one_hot(torch.tensor(idx), num_classes=vocab_size).float()
            if one_hot
            else torch.tensor([idx])
        )
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        result.append(idx_to_char[idx])
    return "".join(result)


def generate_context(
    model,
    char_to_idx: dict,
    idx_to_char: dict,
    context_size: int = 3,
    length: int = 300,
    seed: str = "\n",
    one_hot: bool = False,
) -> str:
    """Генерира текст от модел с контекстен прозорец.

    model        — nn.Module, приема тензор (B, context_size) и връща логити.
    context_size — колко символа контекст очаква моделът.
    one_hot      — ако True, кодира входа като конкатенирани one-hot вектори.
    seed         — начални символи; ако е по-кратък от context_size, се допълва с '\\n'.
    """
    vocab_size = len(char_to_idx)
    # Pad seed to context_size
    if len(seed) < context_size:
        seed = "\n" * (context_size - len(seed)) + seed
    seed = seed[-context_size:]
    context = [char_to_idx[c] for c in seed]
    result = list(seed)

    for _ in range(length):
        x = torch.tensor([context])  # (1, context_size)
        if one_hot:
            x = encode_context(x, vocab_size)  # (1, context_size * vocab_size)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        result.append(idx_to_char[idx])
        context = context[1:] + [idx]

    return "".join(result)
