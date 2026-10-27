# Ek (a): data loader + weight tying + GPT-2 başlangıç değerleri + metnin tamamında eğitim.

import os
import sys
import time
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-2"))
from gpt2 import GPT, GPTConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"            # sorun çıkarırsa "cpu" yaz
else:
    device = "cpu"
print(f"cihaz: {device}")

enc = tiktoken.get_encoding("gpt2")

# ------------------------------------------------------------------
class DataLoaderLite:
    """Metnin tamamını tokenlaştırır, sırayla B*T'lik batch'ler verir, sona gelince başa döner."""

    def __init__(self, path, B, T):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        self.B, self.T = B, T
        self.pos = 0
        self.batches_per_epoch = len(self.tokens) // (B * T)
        print(f"{len(self.tokens):,} token yüklendi | 1 epoch = {self.batches_per_epoch} batch (B={B}, T={T})")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos + B * T + 1]    # +1: son token'ın hedefi için
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T
        if self.pos + B * T + 1 > len(self.tokens):          # sonraki batch sığmıyorsa başa dön
            self.pos = 0
        return x, y

# ------------------------------------------------------------------
class GPTEk(GPT):
    """Görev 2'deki GPT + (isteğe bağlı) GPT-2 başlangıç değerleri + weight tying."""

    def __init__(self, config, tie_weights=True, gpt2_init=True):
        super().__init__(config)
        if gpt2_init:
            self.apply(self._init_weights)
            # residual yoluna yazan projeksiyonlar: her blokta 2 katkı var -> 1/sqrt(2*n_layer)
            std = 0.02 * (2 * config.n_layer) ** -0.5
            for name, p in self.named_parameters():
                if name.endswith("attn.c_proj.weight") or name.endswith("mlp.c_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=std)
        if tie_weights:
            self.transformer.wte.weight = self.lm_head.weight   # tek tensör, iki isim

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

def n_params(model):
    return sum(p.numel() for p in model.parameters())      # paylaşılan tensör bir kez sayılır

@torch.no_grad()
def generate(model, idx, max_new_tokens, top_k=50):
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        idx = torch.cat([idx, torch.gather(topk_idx, -1, ix)], dim=1)
    return idx

# ------------------------------------------------------------------
cfg = GPTConfig()
B, T = 4, 128
loader = DataLoaderLite(os.path.join(HERE, "input.txt"), B, T)

print("\n" + "=" * 70)
print("1) PARAMETRE SAYISI: weight tying ne kazandırdı?")
print("=" * 70)
torch.manual_seed(1337)
m_untied = GPTEk(cfg, tie_weights=False, gpt2_init=False)
torch.manual_seed(1337)
m_tied = GPTEk(cfg, tie_weights=True, gpt2_init=True)
a, b = n_params(m_untied), n_params(m_tied)
print(f"ayrı lm_head ile : {a:,}")
print(f"tying ile        : {b:,}")
print(f"kazanç           : {a - b:,}   (50257 x 768 = {cfg.vocab_size * cfg.n_embd:,})")
print(f"ayrı modelin %{(a - b) / a * 100:.1f}'i   |   tied modelin %{(a - b) / b * 100:.1f}'i")
print(f"wte ve lm_head aynı tensör mü: "
      f"{m_tied.transformer.wte.weight.data_ptr() == m_tied.lm_head.weight.data_ptr()}")

print("\n" + "=" * 70)
print("2) BAŞLANGIÇ DEĞERLERİ: ilk loss ve ağırlık std'leri")
print("=" * 70)
x, y = loader.next_batch()
x, y = x.to(device), y.to(device)
loader.pos = 0                                              # eğitim aynı yerden başlasın
for ad, m in (("varsayılan, ayrı (Görev 4'teki)", m_untied), ("GPT-2 init + tying", m_tied)):
    m.to(device).eval()
    with torch.no_grad():
        logits, loss = m(x, y)
    std = lambda t: t.float().std().item()
    cproj = m.transformer.h[0].attn.c_proj.weight
    print(f"\n{ad}")
    print(f"  ilk loss {loss.item():.4f}  (ln(50257) = 10.8249)   logits varyansı {logits.var().item():.4f}")
    print(f"  std: wte {std(m.transformer.wte.weight):.4f} | wpe {std(m.transformer.wpe.weight):.4f} | "
          f"c_proj {std(cproj):.4f}")
m_untied.cpu()
del m_untied

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) METNİN TAMAMINDA EĞİTİM (tied + GPT-2 init)")
print("=" * 70)
model = m_tied.to(device).train()
n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else loader.batches_per_epoch
print(f"adım sayısı: {n_steps}")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

t_start = time.time()
for step in range(n_steps):
    t0 = time.time()
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    loss_val = loss.item()
    dt = time.time() - t0
    if step % 20 == 0 or step == n_steps - 1:
        print(f"adım {step:4d} | loss {loss_val:.4f} | {dt * 1000:7.1f} ms | {B * T / dt:8.0f} token/sn")
print(f"toplam süre: {(time.time() - t_start) / 60:.1f} dk")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) ÖRNEK ÜRETİM (\"First Citizen:\" ile başla, top-k=50)")
print("=" * 70)
model.eval()
torch.manual_seed(42)
prompt = torch.tensor([enc.encode("First Citizen:\n")], device=device).repeat(2, 1)
out = generate(model, prompt, 60)
for i in range(2):
    print(f"[{i + 1}] {enc.decode(out[i].tolist())!r}")