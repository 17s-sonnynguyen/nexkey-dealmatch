import numpy as np
import torch
from .text_builders import tokenize

def encode_text_dual(bundle, text: str, max_len: int = 48):
    ids = [bundle.dual_vocab.get(w, bundle.UNK_ID_DUAL) for w in tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids += [bundle.PAD_ID_DUAL] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

def encode_pair_cross(bundle, query_text: str, deal_text: str, max_len: int = 96):
    q_ids = [bundle.cross_vocab.get(w, bundle.UNK_ID) for w in tokenize(query_text)]
    d_ids = [bundle.cross_vocab.get(w, bundle.UNK_ID) for w in tokenize(deal_text)]

    q_max = int(max_len * 0.45)
    d_max = max_len - q_max - 1

    q_ids = q_ids[:q_max]
    d_ids = d_ids[:d_max]

    ids = q_ids + [bundle.SEP_ID] + d_ids
    if len(ids) < max_len:
        ids += [bundle.PAD_ID] * (max_len - len(ids))

    ids = np.array(ids, dtype=np.int64)
    ids = np.clip(ids, 0, len(bundle.cross_vocab) - 1)
    return ids

def retrieve_top_n(bundle, prompt: str, top_n: int = 50, max_len: int = 48):
    q_ids = torch.tensor(encode_text_dual(bundle, prompt, max_len=max_len), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        q_vec = bundle.dual.query_encoder(q_ids).cpu().numpy()
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)

    sims = (bundle.deal_vecs @ q_vec.T).squeeze(1)
    idx = np.argsort(-sims)[:top_n]
    return idx, sims[idx]

def rerank(bundle, prompt: str, top_n: int = 50, top_k: int = 5, max_len_cross: int = 96):
    idx, sims = retrieve_top_n(bundle, prompt, top_n=top_n)
    batch = [encode_pair_cross(bundle, prompt, bundle.properties.iloc[i]["deal_text"], max_len=max_len_cross) for i in idx]

    X = torch.tensor(np.stack(batch), dtype=torch.long)
    with torch.no_grad():
        logits = bundle.cross(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    expected_rel = (probs * np.array([0,1,2,3], dtype=np.float32)).sum(axis=1)
    order = np.argsort(-expected_rel)[:top_k]
    final_idx = idx[order]

    out = bundle.properties.iloc[final_idx].copy()
    out["retrieval_sim"] = sims[order]
    out["rerank_score"] = expected_rel[order]
    return out
