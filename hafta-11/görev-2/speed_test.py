# Görev 2: Eğitim adımını hızlandır ve her adımın kazancını ölç.

import gc
import os
import statistics
import sys
import time
import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "..", "görev-1")           # gpt2.py ve input.txt burada
sys.path.append(G1)
from gpt2 import GPT, GPTConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"cihaz: {device}")

B, T = 4, 256
WARMUP, ITERS = 3, 10
KAZANC_ESIGI = 0.97          # bir adım süreyi en az %3 azaltmazsa tutulmaz
use_compile = "--compile" in sys.argv

enc = tiktoken.get_encoding("gpt2")
with open(os.path.join(G1, "input.txt"), "r", encoding="utf-8") as f:
    tokens = torch.tensor(enc.encode(f.read()))

def get_batch(i):
    start = i * B * T
    buf = tokens[start:start + B * T + 1]
    return buf[:-1].view(B, T).to(device), buf[1:].view(B, T).to(device)

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

SDPA_ORIG = F.scaled_dot_product_attention

def manual_attention(q, k, v, is_causal=True, **kw):
    """Hafta 8'deki gibi elle yazılmış attention: (T x T) matris belleğe yazılıyor."""
    Tq = q.size(-2)
    att = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    mask = torch.tril(torch.ones(Tq, Tq, device=q.device)).view(1, 1, Tq, Tq)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v

def measure(cfg):
    """(süre listesi, None) ya da (None, hata mesajı) döndürür."""
    F.scaled_dot_product_attention = SDPA_ORIG if cfg["sdpa"] else manual_attention
    model = opt = fwd = None
    times = []
    try:
        torch.manual_seed(1337)
        model = GPT(GPTConfig(vocab_size=cfg["vocab"])).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        fwd = torch.compile(model) if cfg["compile"] else model
        for i in range(WARMUP + ITERS):
            x, y = get_batch(i)
            sync()
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            if cfg["amp"] is not None:
                with torch.autocast(device_type=device, dtype=cfg["amp"]):
                    _, loss = fwd(x, y)
            else:
                _, loss = fwd(x, y)
            loss.backward()
            opt.step()
            sync()
            dt = time.time() - t0
            if i >= WARMUP:
                times.append(dt)
        return times, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"
    finally:
        F.scaled_dot_product_attention = SDPA_ORIG
        del model, opt, fwd
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

def ozet(times):
    m = statistics.mean(times)
    return m * 1000, statistics.stdev(times) * 1000, B * T / m

print(f"\nB={B}, T={T} ({B * T} token/adım), {WARMUP} ısınma + {ITERS} ölçülen adım")
print(f"{'adım':<40}{'ms/adım':>12}{'token/sn':>10}   durum")
print("-" * 92)

cur = dict(amp=None, compile=False, sdpa=False, vocab=50257)
times, err = measure(cur)
if err:
    print(f"baz ölçüm başarısız: {err}")
    sys.exit(1)
cur_ms, sd, tps = ozet(times)
base_ms = cur_ms
print(f"{'0) baz: fp32 + elle attention':<40}{cur_ms:>8.1f}±{sd:<3.0f}{tps:>10.0f}   başlangıç", flush=True)

# --- 1) TF32: sadece CUDA (Ampere+) ---
if device == "cuda":
    torch.set_float32_matmul_precision("high")
    times, err = measure(cur)
    ms, sd, tps = ozet(times)
    durum = "tutuldu" if ms < cur_ms * KAZANC_ESIGI else "tutulmadı"
    print(f"{'1) + TF32':<40}{ms:>8.1f}±{sd:<3.0f}{tps:>10.0f}   {durum}", flush=True)
    if ms < cur_ms * KAZANC_ESIGI:
        cur_ms = ms
else:
    print(f"{'1) + TF32':<40}{'-':>12}{'-':>10}   ATLANDI: NVIDIA Ampere+ tensor core'una özgü, {device}'de yok",
          flush=True)

adimlar = [
    ("2) + bfloat16 autocast", dict(amp=torch.bfloat16)),
    ("3) + float16 autocast", dict(amp=torch.float16)),
    ("4) + torch.compile", dict(compile=True)),
    ("5) + SDPA (flash attention yerine)", dict(sdpa=True)),
    ("6) + vocab 50304", dict(vocab=50304)),
]

for ad, degisim in adimlar:
    if degisim.get("compile") and not use_compile:
        print(f"{ad:<40}{'-':>12}{'-':>10}   ATLANDI: --compile ile çalıştırılmadı", flush=True)
        continue
    yeni = {**cur, **degisim}
    times, err = measure(yeni)
    if err:
        print(f"{ad:<40}{'-':>12}{'-':>10}   ATLANDI (hata): {err}", flush=True)
        continue
    ms, sd, tps = ozet(times)
    if ms < cur_ms * KAZANC_ESIGI:
        durum = f"tutuldu ({cur_ms / ms:.2f}x)"
        cur, cur_ms = yeni, ms
    else:
        durum = f"tutulmadı (önceki: {cur_ms:.1f} ms)"
    print(f"{ad:<40}{ms:>8.1f}±{sd:<3.0f}{tps:>10.0f}   {durum}", flush=True)

print("-" * 92)
print(f"toplam: {base_ms:.1f} ms -> {cur_ms:.1f} ms  ({base_ms / cur_ms:.2f}x)")
print(f"son ayar: {cur}")

"""
- Senin donanımında hangi adım en çok kazandırdı? -
Donanımım Apple Silicon (mps), bu yüzden TF32'yi atladım,
çünkü NVIDIA Ampere ve sonrasına özgü. Testi iki kez çalıştırdım.
Her iki koşuda da bfloat16 (yaklaşık %20–25) ve torch.compile 
yaklaşık %40) beni yavaşlattı. Nedenlerini doğrulamadım, mps’de bu
yolların henüz olgun olmaması olası bir açıklama. float16 autocast
birinci koşuda 850 ms’den 703 ms’ye (1.21x) düşürdü ama ikincisinde
fark yoktu (804 → 787 ms). SDPA (1.09x) ve vocab’ı 50304'e yuvarlama
(1.06x) ikinci koşuda kazandırdı ama birincisinde kazandırmadı.
Aynı baz ölçüm bile iki koşuda %6 değiştiği için bu küçük farkları
gerçek bir hızlanma olarak yorumlamıyorum. Benim donanımımda toplam
kazanç en fazla yaklaşık 1.2 kat. Videodaki büyük hızlanmalar A100
GPU üzerindeydi, benimkinde aynı sonucu görmedim.
"""