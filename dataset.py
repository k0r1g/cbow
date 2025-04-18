# dataset.py – only class definitions
# --------------------------------------------------
# Minimal dataset module that exposes *only* torch
# Dataset classes (no executable code, no helper
# functions, no tests). Import it anywhere without
# side‑effects.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import torch

__all__ = [
    "HNTitles",
    "HNTitlesWithScore",
]

# ---------------------------------------------------------------------------
# Utility to load a pickle; keeps code DRY but stays internal
# ---------------------------------------------------------------------------

def _load_pickle(path: str | Path):
    return pickle.load(open(Path(path), "rb"))


# ---------------------------------------------------------------------------
# 1️⃣  Hacker News titles – variable‑length token sequences
# ---------------------------------------------------------------------------
class HNTitles(torch.utils.data.Dataset):
    """Iterable of padded / unpadded token‑ID sequences for each HN title."""

    def __init__(self, tokens_path: str | Path = "title_token_ids.pkl"):
        self.title_ids: List[List[int]] = _load_pickle(tokens_path)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.title_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(self.title_ids[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# 2️⃣  (tokens, score) pairs – for regression tasks
# ---------------------------------------------------------------------------
class HNTitlesWithScore(torch.utils.data.Dataset):
    """Dataset yielding (token_ids, score) tuples."""

    def __init__(
        self,
        tokens_path: str | Path = "title_token_ids.pkl",
        scores_path: str | Path = "scores.pkl",
    ):
        self.title_ids: List[List[int]] = _load_pickle(tokens_path)
        self.scores: torch.Tensor = torch.tensor(_load_pickle(scores_path), dtype=torch.float32)
        assert len(self.title_ids) == len(self.scores), "tokens & scores length mismatch"

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.scores)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        ids = torch.tensor(self.title_ids[idx], dtype=torch.long)
        score = self.scores[idx].unsqueeze(0)  # keep shape (1,)
        return ids, score
