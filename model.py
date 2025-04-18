# model.py
#
# Stand‑alone model definitions used by the training / fine‑tuning scripts.
# --------------------------------------------------------------------------
import torch
from gensim.models import Word2Vec
from pathlib import Path


# --------------------------------------------------------------------------
# CBOW – backed by a *pre‑trained / fine‑tuned* Gensim Word2Vec model
# --------------------------------------------------------------------------
class CBOW(torch.nn.Module):
    """
    Wrap a gensim.models.Word2Vec so that it looks exactly like the
    CBOW network your Torch training code expects:
      • self.emb – nn.Embedding initialised with the Word2Vec vectors
      • self.ffw – linear layer sharing the same weights (weight‑tying)
      • forward(input_ids) → logits over the vocabulary
    """
    def __init__(self, gensim_model_path: str | Path, trainable: bool = False):
        """
        Args
        ----
        gensim_model_path : str | Path
            Path to the *.model* file produced by your
            `finetune_hn_word2vec.py` script.
        trainable : bool, default False
            If True the embeddings stay *trainable* in downstream tasks.
            If False they are frozen.
        """
        super().__init__()

        # --- 1.  Load the Word2Vec model ----------------------------------
        self._w2v: Word2Vec = Word2Vec.load(str(gensim_model_path))
        vectors = self._w2v.wv.vectors                      # (V, D)
        vocab_size, emb_dim = vectors.shape

        # --- 2.  Create nn.Embedding initialised with the W2V matrix ------
        self.emb = torch.nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim)
        self.emb.weight.data.copy_(torch.from_numpy(vectors))
        self.emb.weight.requires_grad = trainable           # freeze / unfreeze

        # --- 3.  Weight‑tied projection head ------------------------------
        self.ffw = torch.nn.Linear(in_features=emb_dim,
                                   out_features=vocab_size,
                                   bias=False)
        # **Tie** the weights so ffw.weight == emb.weight
        self.ffw.weight = self.emb.weight

        # --- 4.  Convenience lookup tables --------------------------------
        # gensim already stores these, but having explicit dicts is handy
        self.vocab_to_int: dict[str, int] = {
            word: idx for idx, word in enumerate(self._w2v.wv.index_to_key)
        }
        self.int_to_vocab: list[str] = self._w2v.wv.index_to_key

    # ----------------------------------------------------------------------
    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inpt : LongTensor of shape (batch, context_window)
            Context word‑IDs for the CBOW prediction task.

        Returns
        -------
        logits : FloatTensor of shape (batch, vocab_size)
        """
        emb = self.emb(inpt)              # (B, ctx, D)
        emb = emb.mean(dim=1)             # (B, D)
        logits = self.ffw(emb)            # (B, V)
        return logits

    # ----------------------------------------------------------------------
    # Extra utility – not used by the training loop but often convenient
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def vector(self, word: str) -> torch.Tensor:
        """Retrieve a normalised embedding vector for *word*."""
        idx = self.vocab_to_int[word]
        vec = self.emb.weight[idx]
        return torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze()


# --------------------------------------------------------------------------
# Score‑prediction MLP (unchanged)
# --------------------------------------------------------------------------
class Regressor(torch.nn.Module):
    """
    MLP that consumes an embedding (or pooled sequence embedding)
    and predicts a single scalar (e.g. Hacker News score).
    """
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.seq(inpt)


# --------------------------------------------------------------------------
# Quick smoke‑test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Expect a trained *.model* file path as a CLI arg; fall back to dummy
    import sys, tempfile
    path = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkstemp(suffix=".model")[1]

    try:
        cbow = CBOW(path, trainable=False)
        print("Loaded CBOW with",
              cbow.emb.num_embeddings, "tokens and",
              cbow.emb.embedding_dim, "dimensions.")
    except Exception as exc:
        print("⚠️  Could not load gensim model:", exc)

    # Dummy forward pass (just to prove the API)
    dummy_ids = torch.randint(0, 10, (4, 5))  # batch=4, ctx=5
    if hasattr(cbow, "emb"):
        logits = cbow(dummy_ids)
        print("Logits shape:", logits.shape)

    reg = Regressor(emb_dim=cbow.emb.embedding_dim if hasattr(cbow, "emb") else 128)
    print("Regressor OK – params:", sum(p.numel() for p in reg.parameters()))
