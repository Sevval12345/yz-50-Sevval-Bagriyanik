# Ek görev-b: GPT-2'nin Türkçe ve İngilizce cümlelerdeki loss'u.

import math
import os
import sys
import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-2"))
from gpt2 import GPT

device = "cpu"
model = GPT.from_pretrained("gpt2").to(device).eval()
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                      # <|endoftext|> = 50256

# (Türkçe, İngilizce) çiftleri: aynı anlam, benzer içerik
CIFTLER = [
    ("Dün arkadaşlarımla birlikte kütüphaneye yürüdüm.",
     "Yesterday I walked to the library with my friends."),
    ("Yaz boyunca okuduğumuz kitaplardan söz ettik.",
     "We talked about the books we had read during the summer."),
    ("Kütüphaneci bize okuma salonunun gece yarısına kadar açık olacağını söyledi.",
     "The librarian told us that the reading room would be open until midnight."),
    ("Sonunda çıktığımızda sokaklar sessizdi ve yıldızlar parlıyordu.",
     "When we finally left, the streets were quiet and the stars were bright."),
    ("Yarın sabah erkenden kalkıp denize gideceğiz.",
     "Tomorrow morning we will get up early and go to the sea."),
    ("Türkiye'nin başkenti Ankara'dır.",
     "The capital of Turkey is Ankara."),
]

def token_nll(text):
    """Cümlenin her token'ı için -log p(token | öncesi). <|endoftext|> bağlamı başlatır."""
    ids = [EOT] + enc.encode(text)
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor(ids[1:], device=device)
    with torch.no_grad():
        logits, _ = model(x)
    logp = F.log_softmax(logits[0], dim=-1)                       # (T, vocab)
    nll = -logp.gather(1, y.unsqueeze(1)).squeeze(1)              # (T,)
    return y.tolist(), nll

def ozet(text):
    toks, nll = token_nll(text)
    toplam = nll.sum().item()
    n_tok, n_char, n_byte = len(toks), len(text), len(text.encode("utf-8"))
    return {"toplam": toplam, "tok": n_tok, "char": n_char, "byte": n_byte,
            "loss_tok": toplam / n_tok,
            "nat_char": toplam / n_char,
            "bit_byte": toplam / (n_byte * math.log(2))}

# ------------------------------------------------------------------
print("=" * 86)
print("1) CÜMLE ÇİFTLERİ")
print("=" * 86)
print(f"{'':<4}{'dil':<5}{'token':>6}{'loss/token':>12}{'nat/karakter':>14}{'bit/byte':>10}   cümle")
print("-" * 86)
toplamlar = {"TR": [0.0, 0, 0, 0], "EN": [0.0, 0, 0, 0]}
for i, (tr, en) in enumerate(CIFTLER, 1):
    for dil, metin in (("TR", tr), ("EN", en)):
        s = ozet(metin)
        t = toplamlar[dil]
        t[0] += s["toplam"]; t[1] += s["tok"]; t[2] += s["char"]; t[3] += s["byte"]
        print(f"{i:<4}{dil:<5}{s['tok']:>6}{s['loss_tok']:>12.3f}{s['nat_char']:>14.3f}{s['bit_byte']:>10.3f}   {metin[:44]}")
    print()

# ------------------------------------------------------------------
print("=" * 86)
print("2) TÜM ÇİFTLERİN TOPLAMI")
print("=" * 86)
print(f"{'dil':<5}{'token':>7}{'loss/token':>12}{'nat/karakter':>14}{'bit/byte':>10}")
print("-" * 48)
sonuc = {}
for dil, (nll, n_tok, n_char, n_byte) in toplamlar.items():
    sonuc[dil] = (nll / n_tok, nll / n_char, nll / (n_byte * math.log(2)))
    print(f"{dil:<5}{n_tok:>7}{sonuc[dil][0]:>12.3f}{sonuc[dil][1]:>14.3f}{sonuc[dil][2]:>10.3f}")
print("-" * 48)
print(f"TR/EN oranı:     loss/token {sonuc['TR'][0] / sonuc['EN'][0]:.2f}x   "
      f"nat/karakter {sonuc['TR'][1] / sonuc['EN'][1]:.2f}x   bit/byte {sonuc['TR'][2] / sonuc['EN'][2]:.2f}x")

# ------------------------------------------------------------------
print("\n" + "=" * 86)
print("3) TOKEN TOKEN: hangi parçalar pahalı? (1. çift)")
print("=" * 86)
for dil, metin in (("TR", CIFTLER[0][0]), ("EN", CIFTLER[0][1])):
    toks, nll = token_nll(metin)
    print(f"\n{dil}: {metin}")
    print(f"  {'token':<16}{'NLL':>7}")
    for t, v in zip(toks, nll.tolist()):
        print(f"  {enc.decode_single_token_bytes(t)!r:<16}{v:>7.2f}")