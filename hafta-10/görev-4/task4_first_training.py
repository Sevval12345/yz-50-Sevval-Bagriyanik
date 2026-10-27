# Görev 4: İlk eğitim adımları.
#   1) Rastgele başlatılmış modelin ilk loss'u yaklaşık ln(50257) mi?
#   2) Tek bir batch'i ezberlet: loss sıfıra yaklaşmalı.
# input.txt aynı klasörde olmalı!

import math
import os
import sys
import time
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-2"))
from gpt2 import GPT, GPTConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"          # sorun çıkarırsa "cpu" yaz
else:
    device = "cpu"
print(f"cihaz: {device}")

# ------------------------------------------------------------------
# Veri: Tiny Shakespeare'in başından tek bir batch (B=4, T=32)
enc = tiktoken.get_encoding("gpt2")
corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "input.txt")
with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

B, T = 4, 32
tokens = enc.encode(text[:2000])                    # ilk 2000 karakter, B*T+1 token'dan fazla
buf = torch.tensor(tokens[:B * T + 1])
x = buf[:-1].view(B, T).to(device)                  # girdi
y = buf[1:].view(B, T).to(device)                   # hedef = girdinin 1 kaydırılmışı
print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}")
print(f"x[0] ilk 8 token: {enc.decode(x[0, :8].tolist())!r}")
print(f"y[0] ilk 8 token: {enc.decode(y[0, :8].tolist())!r}   (x'in bir kaydırılmışı)")

V = GPTConfig().vocab_size

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("1) RASTGELE BAŞLATILMIŞ MODELİN İLK LOSS'U")
print("=" * 70)
torch.manual_seed(1337)
model = GPT(GPTConfig()).to(device)
with torch.no_grad():
    logits, loss = model(x, y)
beklenen = math.log(V)
print(f"ilk loss                : {loss.item():.4f}")
print(f"ln({V}) (düzgün dağılım): {beklenen:.4f}")
print(f"fark                    : {loss.item() - beklenen:+.4f}")
print(f"logits varyansı         : {logits.var().item():.4f}")
print(f"ln(V) + varyans/2 tahmini: {beklenen + logits.var().item() / 2:.4f}   (loss buna yakın olmalı)")

# Karşılaştırma için: eğitilmiş GPT-2 aynı batch'te
pretrained = GPT.from_pretrained("gpt2").to(device).eval()
with torch.no_grad():
    _, loss_pre = pretrained(x, y)
print(f"\nkıyas: eğitilmiş GPT-2 aynı batch'te loss = {loss_pre.item():.4f}")
del pretrained

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) TEK BİR BATCH'İ EZBERLET (aynı x, y ile 300 adım)")
print("=" * 70)
torch.manual_seed(1337)
model = GPT(GPTConfig()).to(device)                 # sıfırdan, ilk loss yine aynı
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

n_steps = 300
for i in range(n_steps):
    t0 = time.time()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    loss_val = loss.item()                          # .item() cihazı senkronlar, süre doğru ölçülür
    dt = (time.time() - t0) * 1000
    if i % 20 == 0 or i == n_steps - 1:
        print(f"adım {i:4d} | loss {loss_val:.6f} | {dt:7.1f} ms/adım")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) EZBERLEDİ Mİ? İlk 8 token'dan greedy devam ettir")
print("=" * 70)
model.eval()
with torch.no_grad():
    idx = x[:1, :8].clone()
    for _ in range(T - 8):
        logits, _ = model(idx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)    # sampling yok, en olası token
        idx = torch.cat([idx, nxt], dim=1)
print(f"gerçek: {enc.decode(x[0].tolist())!r}")
print(f"model : {enc.decode(idx[0].tolist())!r}")
print(f"birebir aynı mı: {torch.equal(idx[0], x[0])}")

"""
- Bu test ne işe yarıyır? -

Eğitilmemiş modelin hangi token’ı seçeceğine dair bir fikri yok,
yani 50,257 token’a yaklaşık eşit olasılık vermeli. Doğru token’a 1/50257
olasılık veren modelin loss’u -ln(1/50257) = ln(50257) yaklaşık 10.82.
Benim modelimde 10.87 çıktı, yani başlangıç gerçekten neredeyse
düzgün dağılım. Çok daha düşük çıksaydı bir sızıntı olurdu, çok
daha yüksek çıksaydı model rastgele başlangıçta bile aşırı emin ama yanlış olurdu.

Tek batch’i ezberletme testi şunu soruyor: 128 token’ı ezberleyemeyen
bir model gerçek veride de öğrenemez. Loss 300 adımda 10.87'den 0.0007'ye
düştü ve model ilk 8 token’dan batch’in geri kalanını harfi harfine üretti.
Bu, loss, gradyan akışı ve optimizer’ın çalıştığını gösteriyor. Genelleme hakkında bir şey söylemiyor.”
"""