# Ek Görev-b: Türkçe metinde iki tokenizer, aynı model, aynı süre.

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

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "..", "görev-1")          # gpt2.py
H9 = os.path.join(HERE, "..", "..", "hafta-9")    # tokenizer.py, .model, metin
sys.path.append(G1)
sys.path.append(H9)
from gpt2 import GPT, GPTConfig
from tokenizer import BasicTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--sure", type=int, default=480, help="her koşunun eğitim süresi (sn)")
ap.add_argument("--sira", default="gpt2,benim", help="koşu sırası")
ap.add_argument("--soguma", type=int, default=60, help="koşular arası bekleme (sn)")
ap.add_argument("--metin", default=os.path.join(H9, "turkce_metin2.txt"))
ap.add_argument("--model", default=os.path.join(H9, "turkce_bpe_512.model"))
args = ap.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"cihaz: {device}")

# ---- ayarlar (iki koşuda da aynı) ----
B, T = 8, 256
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384      # süre kısa olduğu için küçük gövde
MAX_LR, MIN_LR, WARMUP, CLIP = 6e-4, 6e-5, 20, 1.0
EVAL_EVERY = 60                           # eğitim süresi saniyesi (eval süresi hariç)
LN2 = math.log(2)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

class GPTEk(GPT):
    """GPT-2 init + weight tying (Hafta 10 Ek (a) ile aynı)."""

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

# ------------------------------------------------------------------
# Metin ve bölme: ilk %90 train, son %10 validation (iki tokenizer için aynı karakterler)
with open(args.metin, "r", encoding="utf-8") as f:
    text = f.read()
cut = int(0.9 * len(text))
train_text, val_text = text[:cut], text[cut:]
print(f"metin: {len(text):,} karakter, {len(text.encode('utf-8')):,} byte | "
      f"train {len(train_text):,} / val {len(val_text):,} karakter")

# ---- tokenizer'lar ----
gpt2_enc = tiktoken.get_encoding("gpt2")
my_tok = BasicTokenizer()
my_tok.load(args.model)

TOKENIZERS = {
    "gpt2": dict(
        ad="GPT-2 BPE", vocab=gpt2_enc.n_vocab,
        encode=lambda s: gpt2_enc.encode(s, allowed_special=set()),
        decode=lambda ids: gpt2_enc.decode(ids),
        token_bytes=lambda i: len(gpt2_enc.decode_single_token_bytes(i)),
    ),
    "benim": dict(
        ad="Hafta 9 BPE-512", vocab=len(my_tok.vocab),
        encode=lambda s: my_tok.encode(s),
        decode=lambda ids: my_tok.decode(ids),
        token_bytes=lambda i: len(my_tok.vocab[i]),
    ),
}

def hazirla(anahtar):
    """Metni token'laştır (önbellekli) ve her token'ın byte uzunluğunu çıkar."""
    tk = TOKENIZERS[anahtar]
    cache = os.path.join(HERE, f"cache_{anahtar}.pt")
    d = None
    if os.path.exists(cache):
        d = torch.load(cache)
        if d.get("n_chars") != len(text):
            d = None
    if d is None:
        t0 = time.time()
        print(f"[{tk['ad']}] token'laştırılıyor (bir kez yapılır, önbelleğe yazılır)...", flush=True)
        tr = torch.tensor(tk["encode"](train_text), dtype=torch.long)
        va = torch.tensor(tk["encode"](val_text), dtype=torch.long)
        d = {"train": tr, "val": va, "n_chars": len(text)}
        torch.save(d, cache)
        print(f"[{tk['ad']}] bitti: {time.time() - t0:.0f} sn", flush=True)
    tr, va = d["train"], d["val"]
    uniq = torch.unique(torch.cat([tr, va])).tolist()
    tb = {i: tk["token_bytes"](i) for i in uniq}
    lut = torch.zeros(max(uniq) + 1, dtype=torch.long)
    for i, n in tb.items():
        lut[i] = n
    tr_b, va_b = lut[tr], lut[va]
    # kontrol: token byte uzunluklarının toplamı metnin gerçek byte sayısı mı?
    ok_tr = int(tr_b.sum()) == len(train_text.encode("utf-8"))
    ok_va = int(va_b.sum()) == len(val_text.encode("utf-8"))
    print(f"[{tk['ad']}] train {len(tr):,} token, val {len(va):,} token | "
          f"byte/token (train) {tr_b.sum().item() / len(tr):.3f} | "
          f"byte toplamı metinle tutarlı mı: train {ok_tr}, val {ok_va}")
    assert ok_tr and ok_va, "byte toplamı tutmuyor, bpb hesabı güvenilmez"
    cs = torch.zeros(len(tr) + 1, dtype=torch.long)
    cs[1:] = torch.cumsum(tr_b, 0)                          # train için byte önek toplamı
    return dict(train=tr, val=va, val_bytes=va_b, train_cs=cs, bytes_per_token=tr_b.sum().item() / len(tr))

def lr_at(frac, step):
    if step < WARMUP:
        return MAX_LR * (step + 1) / WARMUP
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0))) * (MAX_LR - MIN_LR)

@torch.no_grad()
def degerlendir(model, veri):
    """Validation'ın tamamında: (token başına loss, byte başına bit)."""
    model.eval()
    va, vb = veri["val"], veri["val_bytes"]
    n = (len(va) - 1) // T
    x = va[: n * T].view(n, T)
    y = va[1: n * T + 1].view(n, T)
    yb = vb[1: n * T + 1].sum().item()                      # hedef token'ların toplam byte'ı
    toplam_nll, sayi = 0.0, 0
    for i in range(0, n, 16):
        xb, ybt = x[i:i + 16].to(device), y[i:i + 16].to(device)
        _, loss = model(xb, ybt)
        toplam_nll += loss.item() * ybt.numel()
        sayi += ybt.numel()
    model.train()
    return toplam_nll / sayi, toplam_nll / LN2 / yb

@torch.no_grad()
def uret(model, tk, prompt="Sabah erkenden", min_byte=400, max_token=400):
    model.eval()
    out = torch.tensor([tk["encode"](prompt)], dtype=torch.long, device=device)
    for _ in range(max_token):
        logits, _ = model(out[:, -T:])
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        tv, ti = torch.topk(probs, 50, dim=-1)
        nxt = ti.gather(-1, torch.multinomial(tv, 1))
        out = torch.cat([out, nxt], dim=1)
        if len(tk["decode"](out[0].tolist()).encode("utf-8")) >= min_byte:
            break
    model.train()
    return tk["decode"](out[0].tolist())

def kos(anahtar):
    tk = TOKENIZERS[anahtar]
    veri = hazirla(anahtar)
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=T, vocab_size=tk["vocab"], n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
    model = GPTEk(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)
    print(f"\n{'=' * 78}\nKOŞU: {tk['ad']} | vocab {tk['vocab']} | {n_param:,} parametre | "
          f"{args.sure} sn eğitim\n{'=' * 78}")

    tr, cs = veri["train"], veri["train_cs"]
    pos, step = 0, 0
    train_sn, bytes_gorulen, tokens_gorulen = 0.0, 0, 0
    sonraki_eval = EVAL_EVERY
    log = []

    def eval_ve_kaydet():
        vl, bpb = degerlendir(model, veri)
        log.append(dict(sn=train_sn, adim=step, bayt=bytes_gorulen, val_loss_token=vl, val_bpb=bpb))
        print(f"  >> EVAL | {train_sn:5.0f} sn | adım {step:5d} | görülen {bytes_gorulen / 1e6:6.2f} MB | "
              f"val loss/token {vl:.4f} | val bpb {bpb:.4f}", flush=True)

    while train_sn < args.sure:
        t0 = time.time()
        if pos + B * T + 1 > len(tr):
            pos = 0
        buf = tr[pos: pos + B * T + 1]
        x, y = buf[:-1].view(B, T).to(device), buf[1:].view(B, T).to(device)
        bayt = (cs[pos + B * T + 1] - cs[pos + 1]).item()
        pos += B * T
        for g in opt.param_groups:
            g["lr"] = lr_at(train_sn / args.sure, step)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sync()
        train_sn += time.time() - t0
        step += 1
        bytes_gorulen += bayt
        tokens_gorulen += B * T
        if step % 25 == 0:
            print(f"adım {step:5d} | {train_sn:5.0f} sn | loss/token {loss.item():.4f} | "
                  f"{(time.time() - t0) * 1000:5.0f} ms/adım", flush=True)
        if train_sn >= sonraki_eval:
            eval_ve_kaydet()
            sonraki_eval += EVAL_EVERY
    eval_ve_kaydet()

    print(f"\n--- {tk['ad']} örnek üretim ('Sabah erkenden', top-k 50) ---")
    print(uret(model, tk))
    print("-" * 60)

    sonuc = dict(anahtar=anahtar, ad=tk["ad"], vocab=tk["vocab"], parametre=n_param,
                 bayt_per_token=veri["bytes_per_token"], adim=step, token=tokens_gorulen,
                 bayt=bytes_gorulen, epoch=bytes_gorulen / int(veri["train_cs"][-1]),
                 ms_adim=args.sure * 1000 / step, son=log[-1], log=log)
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()
    return sonuc

sonuclar = []
sira = [s.strip() for s in args.sira.split(",")]
for i, a in enumerate(sira):
    if i > 0 and args.soguma > 0:
        print(f"\n{args.soguma} sn soğuma bekleniyor...", flush=True)
        time.sleep(args.soguma)
    sonuclar.append(kos(a))

# ------------------------------------------------------------------
print("\n" + "=" * 110)
print(f"ÖZET (aynı süre: {args.sure} sn eğitim, B={B}, T={T})")
print("=" * 110)
print(f"{'tokenizer':<18}{'vocab':>7}{'param':>13}{'bayt/tok':>10}{'adım':>7}{'ms/adım':>9}"
      f"{'görülen MB':>12}{'epoch':>7}{'val loss/tok':>14}{'val bpb':>10}")
for s in sonuclar:
    print(f"{s['ad']:<18}{s['vocab']:>7}{s['parametre']:>13,}{s['bayt_per_token']:>10.3f}{s['adim']:>7}"
          f"{s['ms_adim']:>9.0f}{s['bayt'] / 1e6:>12.2f}{s['epoch']:>7.2f}"
          f"{s['son']['val_loss_token']:>14.4f}{s['son']['val_bpb']:>10.4f}")
print("\nNOT: 'val loss/tok' iki tokenizer için kıyaslanamaz; kıyas 'val bpb' ile yapılır.")

with open(os.path.join(HERE, "sonuclar.json"), "w", encoding="utf-8") as f:
    json.dump(sonuclar, f, ensure_ascii=False, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for s in sonuclar:
    axes[0].plot([l["sn"] for l in s["log"]], [l["val_bpb"] for l in s["log"]], marker="o", label=s["ad"])
    axes[1].plot([l["bayt"] / 1e6 for l in s["log"]], [l["val_bpb"] for l in s["log"]], marker="o", label=s["ad"])
axes[0].set_xlabel("eğitim süresi (sn)")
axes[1].set_xlabel("görülen metin (MB)")
for ax in axes:
    ax.set_ylabel("validation bpb (düşük = iyi)")
    ax.legend()
axes[0].set_title("aynı süre")
axes[1].set_title("aynı miktarda görülen metin")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "tokenizer_karsilastirma.png"), dpi=150)
plt.close()
print("Grafik: tokenizer_karsilastirma.png | Sayılar: sonuclar.json")

"""
- Hangisi metni daha çok görüyor? -

İki tokenizer’ın bayt/token değeri neredeyse aynı (2.125 vs 2.124).
Adım başına ikisi de yaklaşık 4.35 KB metin görüyor, yani bir adımda aynı miktarda metin işleniyor.
Önceki mesajımda GPT-2'nin adım başına daha çok mrtin göreceğini bekliyordum ama yanılmışım.
2.8 kat fark tamamen adım sayısından geliyor (2255 / 809 = 2.79).
Bunun nedeni, 50k vocab’lı modelin adımının yaklaşık 2.8 kat yavaş olması (593 vs 213 ms).
Kabaca hesapla (ölçmedim): lm_head ileri+geri yaklaşık 240 GFLOP, geri kalan gövde yaklaşık 130 GFLOP,
512 vocab’da lm_head yaklaşık 2 GFLOP. Oran yaklaşık 2.8, ölçtüğünle uyuyor. Yani yavaşlığın büyük kısmı
büyük vocab’ın çıkış katmanından geliyor olabilir. Bu hesap bir açıklama önerisi yani kanıt değil.

- Loss’ları nasıl karşılaştırırsın? -

bpb ile. Bu koşuda bayt/token eşit çıktığı için token başına loss ile bpb sıralaması zufen aynı çıkıyor
ama genel olarak ayrı vocab’larda token başına loss kıyaslanmaz. 480 sn sonunda seninki daha iyi (2.32 vs 2.51, yaklaşık %7).

"""