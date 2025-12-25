import json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ..config import DATA_PATH, CKPT_PATH
from .text_builders import property_to_text

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

    def forward(self, token_ids):
        x = self.emb(token_ids)
        mask = (token_ids != self.pad_id).float().unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

class DualEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, pad_id=0):
        super().__init__()
        self.query_encoder = TextEncoder(vocab_size, emb_dim, pad_id=pad_id)
        self.deal_encoder  = TextEncoder(vocab_size, emb_dim, pad_id=pad_id)

class CrossEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, token_ids):
        x = self.emb(token_ids)
        mask = (token_ids != self.pad_id).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.mlp(pooled)

class ModelBundle:
    """
    Loads all artifacts once at API startup.
    """
    def __init__(self):
        # Data
        self.properties = pd.read_csv(DATA_PATH / "properties.csv")
        self.properties["deal_text"] = self.properties.apply(property_to_text, axis=1)

        # Vocabs
        with open(CKPT_PATH / "dual_vocab_v1.json", "r") as f:
            self.dual_vocab = json.load(f)

        cross_vocab_path = CKPT_PATH / "cross_vocab_v1.json"
        if cross_vocab_path.exists():
            with open(cross_vocab_path, "r") as f:
                self.cross_vocab = json.load(f)
        else:
            self.cross_vocab = copy.deepcopy(self.dual_vocab)
            if "<SEP>" not in self.cross_vocab:
                self.cross_vocab["<SEP>"] = len(self.cross_vocab)

        self.PAD_ID_DUAL = self.dual_vocab["<PAD>"]
        self.UNK_ID_DUAL = self.dual_vocab["<UNK>"]

        self.PAD_ID = self.cross_vocab["<PAD>"]
        self.UNK_ID = self.cross_vocab["<UNK>"]
        self.SEP_ID = self.cross_vocab["<SEP>"]

        # Deal embeddings
        self.deal_vecs = np.load(CKPT_PATH / "deal_vecs_v1.npy")
        self.deal_vecs = self.deal_vecs / (np.linalg.norm(self.deal_vecs, axis=1, keepdims=True) + 1e-9)

        # Models
        self.dual = DualEncoder(vocab_size=len(self.dual_vocab), emb_dim=128, pad_id=self.PAD_ID_DUAL)
        self.dual.load_state_dict(torch.load(CKPT_PATH / "dual_encoder_v1.pt", map_location="cpu"))
        self.dual.eval()

        self.cross = CrossEncoder(vocab_size=len(self.cross_vocab), emb_dim=128, hidden=128, pad_id=self.PAD_ID)
        self.cross.load_state_dict(torch.load(CKPT_PATH / "cross_encoder_best.pt", map_location="cpu"))
        self.cross.eval()
