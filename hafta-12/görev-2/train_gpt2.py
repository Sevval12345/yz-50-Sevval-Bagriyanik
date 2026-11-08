# Görev 2: eğitim döngüsüne doğrulama loss'u ve örnek üretim ekle.
# Not: Aynı script Görev 3'te kullanılacak.

import argparse
import json
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-1"))
from data_loader import ShardedDataLoader
from gpt2 import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100, help="adım sayısı (--sure verilmezse)")
ap.add_argument("--sure", type=int, default=None, help="eğitim süresi (sn); verilirse adım sayısı yerine süreye göre çalışır")
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--val-steps", type=int, default=20, help="her eval'de kaç batch")
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--T", type=int, default=256)
ap.add_argument("--total-batch", type=int, default=4096, help="adım başına token (B*T*grad_accum)")
ap.add_argument("--vocab", type=int, default=50304, help="50304: 128'in katı; 50257: orijinal")
ap.add_argument("--max-lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--shards", default=os.path.join(HERE, "..", "görev-1", "fineweb_shards"))
ap.add_argument("--out", default=HERE)
ap.add_argument("--prompt", default="Hello, I'm a language model,")
args = ap.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"cihaz: {device}")

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

B, T = args.B, args.T
assert args.total_batch % (B * T) == 0, "total-batch, B*T'nin katı olmalı"
GRAD_ACCUM = args.total_batch // (B * T)
MAX_LR, MIN_LR, CLIP, WD = args.max_lr, args.max_lr * 0.1, 1.0, 0.1
enc = tiktoken.get_encoding("gpt2")
GERCEK_VOCAB = 50257          # tokenizer'ın gerçek vocab'ı; 50304'te son 47 token kullanılmıyor

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

class GPTEk(GPT):
    """Hafta 10/11'deki gibi: GPT-2 başlangıç değerleri + weight tying."""

    def __init__(self, config):
        super().__init__(config)
        self.apply(self._init_weights)
        std = 0.02 * (2 * config.n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("attn.c_proj.weight") or name.endswith("mlp.c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)
        self.transformer.wte.weight = self.lm_head.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

def get_lr(step, frac):
    """Warmup (adım sayısıyla) + cosine (ilerleme oranı frac ile: adım ya da süre)."""
    if step < args.warmup:
        return MAX_LR * (step + 1) / args.warmup
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * min(max(frac, 0.0), 1.0))) * (MAX_LR - MIN_LR)

def configure_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    print(f"weight decay: {len(decay)} tensör / {sum(p.numel() for p in decay):,} parametre | "
          f"decay yok: {len(nodecay)} tensör / {sum(p.numel() for p in nodecay):,} parametre")
    groups = [{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}]
    kw = {"fused": True} if device == "cuda" else {}
    return torch.optim.AdamW(groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, **kw)

@torch.no_grad()
def val_loss_hesapla(model, val_loader, n_batch):
    """Aynı n_batch doğrulama batch'inin ortalama loss'u (her seferinde val shard'ının başından)."""
    model.eval()
    val_loader.reset()
    toplam = 0.0
    for _ in range(n_batch):
        x, y = val_loader.next_batch()
        _, loss = model(x.to(device), y.to(device))
        toplam += loss.item()
    model.train()
    return toplam / n_batch

@torch.no_grad()
def ornek_uret(model, prompt="Hello, I'm a language model,", n_seq=2, max_new=64, top_k=50, seed=42):
    """Top-k ile kısa metin üretir. Ayrı bir CPU generator kullanır: eğitimin rastgeleliğine dokunmaz."""
    model.eval()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device).repeat(n_seq, 1)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.config.block_size:])
        logits = logits[:, -1, :].clone()
        logits[:, GERCEK_VOCAB:] = float("-inf")           # 50304'teki kullanılmayan token'ları ele
        probs = F.softmax(logits, dim=-1)
        tp, ti = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(tp.cpu(), 1, generator=rng).to(device)
        idx = torch.cat([idx, torch.gather(ti, -1, ix)], dim=1)
    model.train()
    return [enc.decode(satir.tolist()) for satir in idx]

# ------------------------------------------------------------------
train_loader = ShardedDataLoader(args.shards, "train", B, T)
val_loader = ShardedDataLoader(args.shards, "val", B, T)
model = GPTEk(GPTConfig(vocab_size=args.vocab)).to(device)
n_param = sum(p.numel() for p in model.parameters())
optimizer = configure_optimizer(model)
mod = f"{args.sure} sn" if args.sure else f"{args.steps} adım"
print(f"model: {n_param:,} parametre (vocab {args.vocab}) | micro-batch {B}x{T}, accumulation {GRAD_ACCUM}, "
      f"adım başına {args.total_batch:,} token | süre/uzunluk: {mod}")

ornek_dosya = os.path.join(args.out, "ornekler.txt")
open(ornek_dosya, "w", encoding="utf-8").close()

train_losses = []                                        # her adımın loss'u
log = []                                                 # eval kayıtları
step, train_sn, son_eval_adim = 0, 0.0, 0

try:
    while True:
        bitti = (train_sn >= args.sure) if args.sure else (step >= args.steps)

        # ---- doğrulama loss'u + örnek metin ----
        if step % args.eval_every == 0 or bitti:
            vl = val_loss_hesapla(model, val_loader, args.val_steps)
            son = train_losses[son_eval_adim:step]
            tr_ort = sum(son) / len(son) if son else float("nan")
            log.append(dict(adim=step, sn=train_sn, token=step * args.total_batch, val_loss=vl, train_loss_ort=tr_ort))
            print(f">> EVAL | adım {step:5d} | {train_sn:6.0f} sn | val loss {vl:.4f} | train loss (son {len(son)} adım ort.) {tr_ort:.4f}",
                  flush=True)
            metin = ornek_uret(model, prompt=args.prompt)
            with open(ornek_dosya, "a", encoding="utf-8") as f:
                f.write(f"=== adım {step} | val loss {vl:.4f} ===\n")
                for i, m in enumerate(metin):
                    f.write(f"[{i + 1}] {m!r}\n")
                f.write("\n")
            for i, m in enumerate(metin):
                print(f"   [{i + 1}] {m!r}", flush=True)
            son_eval_adim = step
        if bitti:
            break

        # ---- bir eğitim adımı ----
        t0 = time.time()
        frac = train_sn / args.sure if args.sure else (step - args.warmup) / max(1, args.steps - args.warmup)
        lr = get_lr(step, frac)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
            _, loss = model(x.to(device), y.to(device))
            loss = loss / GRAD_ACCUM
            loss_accum += loss.item()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        optimizer.step()
        sync()
        dt = time.time() - t0
        train_sn += dt
        train_losses.append(loss_accum)
        if step % args.log_every == 0:
            print(f"adım {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {norm:6.3f} | "
                  f"{dt * 1000:6.0f} ms | {args.total_batch / dt:6.0f} token/sn", flush=True)
        step += 1
except KeyboardInterrupt:
    print("\nCtrl+C: eğitim durduruldu, o ana kadarki sonuçlar kaydediliyor...")

# ------------------------------------------------------------------
gorulen = step * args.total_batch
print(f"\nbitti: {step} adım, {gorulen:,} token görüldü, ortalama "
      f"{gorulen / max(train_sn, 1e-9):,.0f} token/sn, eğitim süresi {train_sn / 60:.1f} dk")
if log:
    print(f"son val loss: {log[-1]['val_loss']:.4f}")

with open(os.path.join(args.out, "log.json"), "w", encoding="utf-8") as f:
    json.dump(dict(args=vars(args), parametre=n_param, adim=step, token=gorulen, sure_sn=train_sn,
                   train_losses=train_losses, eval=log), f, ensure_ascii=False, indent=1)
torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(train_losses)), train_losses, alpha=0.25, color="tab:blue", label="train (her adım)")
ev = [l for l in log if not math.isnan(l["train_loss_ort"])]
ax.plot([l["adim"] for l in ev], [l["train_loss_ort"] for l in ev], "o-", color="tab:blue",
        label="train (iki eval arası ortalama)")
ax.plot([l["adim"] for l in log], [l["val_loss"] for l in log], "s-", color="tab:orange", label="val")
ax.set_xlabel("adım")
ax.set_ylabel("loss")
ax.set_title("train ve val loss")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out, "loss_egrisi.png"), dpi=150)
plt.close()
print("Kaydedildi: loss_egrisi.png, ornekler.txt, log.json, model.pt")


"""
- Eğitim ilerledikçe ürettiği metin nasıl değişiyor? -

Eğitimin başında model rastgele token’lar üretiyor, anlamsız bir kelime çorbası.
100. adımda artık sadece gerçek İngilizce kelimeler çıkıyor ama çoğu “the, of, it”
gibi sık geçen kelimeler, noktalama da rastgele. 200–300. adımlarda cümleye benzeyen
kalıplar çıkmaya başlıyor (“in the United”, “we can begin of local power”), ama
dilbilgisi hâlâ bozuk. 400–600 arasında konular belirginleşiyor, “child, water, health,
important” gibi eğitim metinlerine özgü kelimeler çok geçiyor. Sonuna doğru tırnak
işaretleri ve <|endoftext|> gibi yapılar da öğreniliyor. Ama bir saatin sonunda bile
üretilen metin anlamlı değil. Kelimeleri ve kalıpları taklit ediyor, ne dediğini bilmiyor.

Bu val loss’la uyumlu: 11'den 6,5'e düştü ama 6,5 hâlâ yüksek. Modelin bir sonraki
kelimeyi tahmin etmedeki kararsızlığı büyük.
"""