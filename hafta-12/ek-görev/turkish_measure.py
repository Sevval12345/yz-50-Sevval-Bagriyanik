# Ek Görev: Türkçe modeli ölç. (a) bits-per-byte, (b) kendi mini "HellaSwag"i.

import math
import os
import random
import sys
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-2"))
from gpt2 import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
GERCEK_VOCAB = 50257

model = GPT(GPTConfig(vocab_size=50304))
model.load_state_dict(torch.load(os.path.join(HERE, "turkce_run", "model.pt"), map_location="cpu"))
model.to(device).eval()
val = np.load(os.path.join(HERE, "turkce_shards", "turkce_val_000.npy")).astype(np.int64)

@torch.no_grad()
def satir_loss(tok):
    """(N, L) token matrisi -> (N, L-1) token başına loss."""
    logits, _ = model(tok.to(device))
    logits = logits[..., :GERCEK_VOCAB].float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, GERCEK_VOCAB), tok[:, 1:].reshape(-1).to(device),
                           reduction="none").view(tok.size(0), -1).cpu()

# ---- (a) tutulmuş metinde loss ve bits-per-byte ----
T = 256
n = (len(val) - 1) // T
toplam, sayi = 0.0, 0
for i in range(0, n, 4):
    parca = torch.stack([torch.from_numpy(val[j * T:j * T + T + 1]) for j in range(i, min(i + 4, n))])
    l = satir_loss(parca)
    toplam += l.sum().item()
    sayi += l.numel()
loss = toplam / sayi
byte_basina_token = len(enc.decode(val.tolist()).encode("utf-8")) / len(val)
bpb = loss / math.log(2) / byte_basina_token
print(f"val loss (T={T}, {n} parça): {loss:.4f} | perplexity: {math.exp(loss):.1f}")
print(f"token başına {byte_basina_token:.2f} byte -> bits-per-byte: {bpb:.3f}")

# ---- (b) mini çoktan seçmeli test (kolay: yanlışlar metnin başka yerlerinden) ----
random.seed(0)
CTX, END, N = 64, 24, 500
dogru = 0
for _ in range(N):
    p = random.randrange(0, len(val) - CTX - END)
    ctx = val[p:p + CTX]
    gercek = val[p + CTX:p + CTX + END]
    adaylar = [gercek]
    while len(adaylar) < 4:
        q = random.randrange(0, len(val) - END)
        if abs(q - (p + CTX)) > END:                  # gerçek devamla çakışmasın
            adaylar.append(val[q:q + END])
    tok = torch.from_numpy(np.stack([np.concatenate([ctx, a]) for a in adaylar]))
    l = satir_loss(tok)[:, CTX - 1:]                  # sadece ending token'larının loss'u
    dogru += int(l.mean(dim=1).argmin().item() == 0)  # 0. aday gerçek devam
print(f"mini test: {dogru}/{N} = {dogru / N:.3f} (rastgele: 0.250)")