# model.py
#
# Stand‑alone model definitions used by the training / fine‑tuning scripts.
# --------------------------------------------------------------------------
import torch
from gensim.models import Word2Vec
from pathlib import Path


# --------------------------------------------------------------------------
# CBOW – backed by a *pre‑trained / fine‑tuned* Gensim Word2Vec model or PyTorch state_dict
# --------------------------------------------------------------------------
class CBOW(torch.nn.Module):
    """
    Wrap a gensim.models.Word2Vec or a PyTorch state_dict so that it looks
    exactly like the CBOW network your Torch training code expects:
      • self.emb – nn.Embedding initialised with the pre-trained vectors
      • self.ffw – linear layer sharing the same weights (weight‑tying)
      • forward(input_ids) → logits over the vocabulary
    """
    def __init__(self, model_path: str | Path, trainable: bool = False, vocab_path: str | Path | None = None):
        """
        Args
        ----
        model_path : str | Path
            Path to the pre-trained model file.
            Can be a *.model* or *.word2vec* file (gensim) or a *.pth* file
            (PyTorch state_dict containing at least 'emb.weight' or similar).
        trainable : bool, default False
            If True the embeddings stay *trainable* in downstream tasks.
            If False they are frozen.
        vocab_path : str | Path | None, default None
            Path to a pickle file containing a tuple `(vocab_to_int, int_to_vocab)`.
            Required if `model_path` is a `.pth` file. Ignored otherwise.
        """
        super().__init__()
        model_path = Path(model_path)

        if model_path.suffix in [".word2vec", ".model"]:
            # --- 1a. Load the Word2Vec model (Gensim) ----------------------
            print(f"Loading Gensim model from: {model_path}")
            self._w2v: Word2Vec = Word2Vec.load(str(model_path))
            vectors = self._w2v.wv.vectors                      # (V, D)
            vocab_size, emb_dim = vectors.shape
            self.vocab_to_int: dict[str, int] = {
                word: idx for idx, word in enumerate(self._w2v.wv.index_to_key)
            }
            self.int_to_vocab: list[str] = self._w2v.wv.index_to_key
            init_weights = torch.from_numpy(vectors)

        elif model_path.suffix == ".pth":
            # --- 1b. Load the PyTorch state dict --------------------------
            print(f"Loading PyTorch state_dict from: {model_path}")
            if vocab_path is None:
                raise ValueError("`vocab_path` is required when loading a .pth state_dict.")
            
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            if 'emb.weight' not in state_dict:
                # Try finding the first embedding layer weights if key is different
                emb_key = None
                for key, tensor in state_dict.items():
                    if tensor.ndim == 2: # Basic check for embedding matrix shape
                        emb_key = key
                        print(f"Warning: 'emb.weight' not found. Using '{emb_key}' as embedding weights.")
                        break
                if emb_key is None:
                     raise KeyError("Could not find embedding weights ('emb.weight' or similar) in the state_dict.")
                init_weights = state_dict[emb_key]
            else:
                init_weights = state_dict['emb.weight']
                
            vocab_size, emb_dim = init_weights.shape

            print(f"Loading vocabulary from: {vocab_path}")
            with open(vocab_path, 'rb') as f:
                loaded_vocab = __import__("pickle").load(f)
                if isinstance(loaded_vocab, tuple) and len(loaded_vocab) == 2:
                     self.vocab_to_int, self.int_to_vocab = loaded_vocab
                     # Ensure int_to_vocab is a list if it was loaded as dict keys or similar
                     if isinstance(self.int_to_vocab, dict):
                         # Recreate list based on indices in vocab_to_int
                         self.int_to_vocab = [""] * len(self.vocab_to_int)
                         for word, idx in self.vocab_to_int.items():
                             if 0 <= idx < len(self.int_to_vocab):
                                 self.int_to_vocab[idx] = word
                         print("Warning: Reconstructed int_to_vocab list from vocab_to_int.")
                     elif not isinstance(self.int_to_vocab, list):
                         raise TypeError("Loaded int_to_vocab is not a list or dict.")
                else:
                     raise TypeError(f"Expected {vocab_path} to contain a tuple (vocab_to_int, int_to_vocab).")
            
            # Validate vocab size match
            if len(self.vocab_to_int) != vocab_size or len(self.int_to_vocab) != vocab_size:
                 print(f"Warning: Loaded vocab size ({len(self.vocab_to_int)}) does not match embedding matrix dimension ({vocab_size}). This might cause issues.")
                 # Adjust vocab_size based on loaded vocab, potentially ignoring some embedding rows
                 vocab_size = len(self.vocab_to_int)

        else:
            raise ValueError(f"Unsupported model file type: {model_path.suffix}. Expected .word2vec, .model, or .pth")

        # --- 2.  Create nn.Embedding initialised with the loaded weights ---
        self.emb = torch.nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim)
        # Ensure weights match the embedding layer size before copying
        self.emb.weight.data.copy_(init_weights[:vocab_size, :]) 
        self.emb.weight.requires_grad = trainable           # freeze / unfreeze

        # --- 3.  Weight‑tied projection head ------------------------------
        self.ffw = torch.nn.Linear(in_features=emb_dim,
                                   out_features=vocab_size,
                                   bias=False)
        # **Tie** the weights so ffw.weight == emb.weight
        self.ffw.weight = self.emb.weight

        # --- 4. Convenience properties ---
        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim

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
    and predicts a single scalar (e.g. Hacker News score).
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
