# Görev 3: GPT-3'ün eğitim ayarları.
#   AdamW (betas 0.9/0.95), gradient clipping + norm, warmup'lı cosine lr,
#   weight decay (sadece 2D+ parametreler), gradient accumulation.

import math
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "..", "görev-1")    # gpt2.py ve input.txt burada
sys.path.append(G1)
from gpt2 import GPT, GPTConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"cihaz: {device}")

# ---- ayarlar ----
B, T = 4, 256                       # micro-batch
TOTAL_BATCH = 4096                  # toplam batch (token)
assert TOTAL_BATCH % (B * T) == 0
GRAD_ACCUM = TOTAL_BATCH // (B * T)
MAX_LR = 6e-4                       # videodaki değer; kararsız çıkarsa 3e-4 dene
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 100
WEIGHT_DECAY = 0.1
CLIP = 1.0

enc = tiktoken.get_encoding("gpt2")

# ------------------------------------------------------------------
def get_lr(it, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS):
    if it < warmup_steps:                                   # 1) doğrusal warmup
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:                                      # 3) bittikten sonra min_lr
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1'den 0'a cosine
    return min_lr + coeff * (max_lr - min_lr)               # 2) cosine decay

class GPTEk(GPT):
    """Hafta 10 Ek (a): GPT-2 başlangıç değerleri + weight tying (görev-1/training.py'den)."""

    def __init__(self, config, tie_weights=True, gpt2_init=True):
        super().__init__(config)
        if gpt2_init:
            self.apply(self._init_weights)
            std = 0.02 * (2 * config.n_layer) ** -0.5
            for name, p in self.named_parameters():
                if name.endswith("attn.c_proj.weight") or name.endswith("mlp.c_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=std)
        if tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

class DataLoaderLite:
    def __init__(self, path, B, T):
        with open(path, "r", encoding="utf-8") as f:
            self.tokens = torch.tensor(enc.encode(f.read()), dtype=torch.long)
        self.B, self.T, self.pos = B, T, 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos + B * T + 1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.pos += B * T
        if self.pos + B * T + 1 > len(self.tokens):
            self.pos = 0
        return x, y

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("1) LEARNING RATE EĞRİSİ (warmup + cosine)")
print("=" * 78)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (ad, mx, wu) in zip(axes, [("videodaki ayar (19,073 adım, warmup 715)", 19073, 715),
                                   (f"bu koşu ({MAX_STEPS} adım, warmup {WARMUP_STEPS})", MAX_STEPS, WARMUP_STEPS)]):
    adimlar = list(range(mx + 1))
    lrs = [get_lr(i, MAX_LR, MIN_LR, wu, mx) for i in adimlar]
    ax.plot(adimlar, lrs)
    ax.axvline(wu, color="gray", linestyle="--", linewidth=1, label=f"warmup sonu ({wu})")
    ax.set_xlabel("adım")
    ax.set_ylabel("learning rate")
    ax.set_title(ad)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "lr_egrisi.png"), dpi=150)
plt.close()
print(f"max_lr {MAX_LR:.0e} | min_lr {MIN_LR:.0e} | lr(0) = {get_lr(0):.2e} | "
      f"lr({WARMUP_STEPS}) = {get_lr(WARMUP_STEPS):.2e} | lr({MAX_STEPS}) = {get_lr(MAX_STEPS):.2e}")
print("Grafik: lr_egrisi.png")

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("2) WEIGHT DECAY GRUPLARI (sadece 2 boyutlu ve üstü parametreler)")
print("=" * 78)

def configure_optimizer(model):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]       # matrisler, embedding'ler
    nodecay = [p for p in params.values() if p.dim() < 2]      # bias, LayerNorm
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in nodecay)
    print(f"decay uygulanan   : {len(decay):>3} tensör, {n_decay:>12,} parametre")
    print(f"decay uygulanmayan: {len(nodecay):>3} tensör, {n_nodecay:>12,} parametre")
    print(f"toplam            : {n_decay + n_nodecay:,}")
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY},
              {"params": nodecay, "weight_decay": 0.0}]
    kw = {"fused": True} if device == "cuda" else {}
    return torch.optim.AdamW(groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, **kw)

torch.manual_seed(1337)
model = GPTEk(GPTConfig()).to(device)
optimizer = configure_optimizer(model)

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("3) GRADIENT ACCUMULATION DOĞRULAMASI (CPU, 8 örnek, T=64)")
print("=" * 78)
torch.manual_seed(1337)
tm = GPTEk(GPTConfig())                                    # test için CPU'da ayrı bir model
xb = torch.randint(0, 50257, (8, 64))
yb = torch.randint(0, 50257, (8, 64))

def grads_of(fn):
    tm.zero_grad(set_to_none=True)
    fn()
    return [p.grad.detach().clone() for p in tm.parameters()]

def norm_of(gs):
    return math.sqrt(sum((g.double() ** 2).sum().item() for g in gs))

def big():
    tm(xb, yb)[1].backward()

def acc(divide):
    def f():
        for xm, ym in zip(xb.chunk(2), yb.chunk(2)):          # 2 micro-batch, 4'er örnek
            loss = tm(xm, ym)[1]
            (loss / 2 if divide else loss).backward()
    return f

g_big = grads_of(big)
g_div = grads_of(acc(True))
g_nodiv = grads_of(acc(False))
n_big, n_div, n_nodiv = norm_of(g_big), norm_of(g_div), norm_of(g_nodiv)
fark = math.sqrt(sum(((a - b).double() ** 2).sum().item() for a, b in zip(g_big, g_div)))
print(f"gradient normu: büyük batch {n_big:.4f} | bölerek biriktir {n_div:.4f} | bölmeden biriktir {n_nodiv:.4f}")
print(f"bölerek biriktirilen ile büyük batch farkı (göreli): {fark / n_big:.2e}")
print(f"bölmeden biriktirilen / büyük batch norm oranı: {n_nodiv / n_big:.3f}   (beklenen: 2, yani adım sayısı)")
del tm, g_big, g_div, g_nodiv

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"4) EĞİTİM: {MAX_STEPS} adım | micro-batch {B}x{T} | accumulation {GRAD_ACCUM} | toplam batch {TOTAL_BATCH} token")
print("=" * 78)
loader = DataLoaderLite(os.path.join(G1, "input.txt"), B, T)
losses, norms, lrs_log = [], [], []
for step in range(MAX_STEPS):
    t0 = time.time()
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss = loss / GRAD_ACCUM                          # her micro-batch ortalamasını N'e böl
        loss_accum += loss.item()
        loss.backward()                                   # gradient'ler birikiyor (+=)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()   # kırpmadan ÖNCEKİ norm
    optimizer.step()
    sync()
    dt = time.time() - t0
    losses.append(loss_accum); norms.append(norm); lrs_log.append(lr)
    print(f"adım {step:3d} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {norm:7.4f} | "
          f"{dt * 1000:7.0f} ms | {TOTAL_BATCH / dt:6.0f} token/sn", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (ad, v) in zip(axes, [("loss", losses), ("gradient norm (kırpma öncesi)", norms), ("learning rate", lrs_log)]):
    ax.plot(v)
    ax.set_title(ad)
    ax.set_xlabel("adım")
axes[1].axhline(CLIP, color="red", linestyle="--", linewidth=1, label=f"clip = {CLIP}")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "egitim_egrileri.png"), dpi=150)
plt.close()
print("\nGrafik: egitim_egrileri.png")

"""
- Loss’u neden accumulation adım sayısına böleriz? -

“Cross-entropy, bir micro-batch içindeki token’ların ortalamasını alıyor.
Büyük batch’i 4 micro-batch’e bölüp her birinin loss’una backward çağırınca
PyTorch gradient’leri toplayarak biriktiriyor. Bölmezsem elimde 4 ortalamanın
toplamı olur. Oysa büyük batch’in doğru gradient’i bütün token’ların
ortalamasından gelir, yani 4 ortalamanın ortalamasından. İkisi arasındaki fark
tam 4 kat. Bu yüzden her micro-batch’in loss’unu 4'e bölüyorum, böylece biriken
gradient büyük batch’in gradient’iyle aynı oluyor.

Bunu kodda da denedşm: 2 micro-batch’le büyük batch’in gradient normu 11.9007
çıktı ve bölerek biriktirilen de 11.9007'ydi (göreli fark 3.8e-7). Bölmeden
biriktirince norm 23.8013 oldu, tam 2 kat.”
"""