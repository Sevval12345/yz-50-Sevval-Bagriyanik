# Görev 3: GPT-2'yi konuştur. Sampling döngüsü + top-k.

import os
import sys
import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "görev-2"))
from gpt2 import GPT

device = "cpu"          # CPU'da seed ile sonuçlar tekrarlanabilir
model = GPT.from_pretrained("gpt2").to(device).eval()
enc = tiktoken.get_encoding("gpt2")

def generate(model, idx, max_new_tokens, top_k=50):
    """
    idx: (B, T) başlangıç token'ları. Her adımda 1 token ekler.
    top_k=None ise tüm vocab'dan çeker (Hafta 8'deki generate gibi).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]          # bağlam 1024'ü aşmasın
        with torch.no_grad():
            logits, _ = model(idx_cond)                       # (B, T, vocab)
        logits = logits[:, -1, :]                             # sadece son pozisyon: (B, vocab)
        probs = F.softmax(logits, dim=-1)
        if top_k is not None:
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)   # (B, top_k)
            ix = torch.multinomial(topk_probs, 1)                      # top-k içinden çek: (B, 1)
            next_token = torch.gather(topk_idx, -1, ix)                # gerçek token id'sine çevir
        else:
            next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx

def konustur(baslangic, n_ornek=5, max_new_tokens=30, top_k=50, seed=42):
    torch.manual_seed(seed)
    ids = enc.encode(baslangic)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(n_ornek, 1).to(device)
    print(f"\nBaşlangıç: {baslangic!r}  ({len(ids)} token)   top_k={top_k}")
    print("-" * 70)
    out = generate(model, idx, max_new_tokens, top_k)
    for i in range(n_ornek):
        print(f"[{i + 1}] {enc.decode(out[i].tolist())!r}")

# ------------------------------------------------------------------
print("=" * 70)
print("1) İNGİLİZCE BAŞLANGIÇ")
print("=" * 70)
konustur("Hello, I'm a language model,")

print("\n" + "=" * 70)
print("2) TÜRKÇE BAŞLANGIÇ")
print("=" * 70)
konustur("Merhaba, ben bir dil modeliyim,")

print("\n" + "=" * 70)
print("3) TOP-K OLMADAN (tüm vocab'dan çek), İngilizce")
print("=" * 70)
konustur("Hello, I'm a language model,", top_k=None)

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) TOKEN SAYISI: GPT-2 tokenizer'ı, Türkçe vs İngilizce")
print("=" * 70)
kisa_tr, kisa_en = "Merhaba, ben bir dil modeliyim.", "Hello, I'm a language model."
for ad, t in [("TR", kisa_tr), ("EN", kisa_en)]:
    ids = enc.encode(t)
    parcalar = [enc.decode_single_token_bytes(i) for i in ids]
    print(f"{ad}: {len(t.split())} kelime, {len(ids)} token")
    print(f"    {parcalar}")

# Geçen haftaki paragraf (tr_en_token_karsilastirma.py'den): GPT-2 vs GPT-4 tokenizer'ı
EN = ("Yesterday I walked from my house to the library with my friends. "
      "We talked about the books we had read during the summer, and then we decided "
      "to study together for the exam. The librarian told us that the reading room "
      "would be open until midnight, so we stayed there for hours. When we finally "
      "left, the streets were quiet and the stars were bright above the city.")
TR = ("Dün arkadaşlarımla birlikte evimden kütüphaneye yürüdüm. "
      "Yaz boyunca okuduğumuz kitaplardan söz ettik, sonra sınav için birlikte "
      "çalışmaya karar verdik. Kütüphaneci bize okuma salonunun gece yarısına kadar "
      "açık olacağını söyledi, bu yüzden orada saatlerce kaldık. Sonunda "
      "çıktığımızda sokaklar sessizdi ve şehrin üzerinde yıldızlar parlıyordu.")

print(f"\n{'tokenizer':<14}{'metin':<10}{'kelime':>7}{'token':>7}{'token/kelime':>14}{'byte/token':>12}")
print("-" * 64)
for enc_ad in ["gpt2", "cl100k_base"]:
    e = tiktoken.get_encoding(enc_ad)
    for ad, t in [("Türkçe", TR), ("İngilizce", EN)]:
        n_tok, n_kel, n_byte = len(e.encode(t)), len(t.split()), len(t.encode("utf-8"))
        print(f"{enc_ad:<14}{ad:<10}{n_kel:>7}{n_tok:>7}{n_tok / n_kel:>14.2f}{n_byte / n_tok:>12.2f}")


"""
- GPT-2'nin Türkçe çıktısı neden kötü? Aynı Türkçe cümle GPT-2'nin tokenizer'ında kaç token tutuyor? Geçen haftaki sonuçlarınla bağla -

1. Veri: Model neredeyse sadece İngilizce metinle eğitildi. Türkçeyi öğrenmesi için yeterli örnek görmedi.
2. Tokenizer: BPE birleştirmelerini yine İngilizce metinden öğrendi.
Türkçe kelimeler Mer|hab|a gibi anlamsız hecelere bölünüyor, ek (-iyim) kökten ayrı parçalanıyor.
Model Türkçe ek yapısını doğrudan göremiyor, bu parçaları ses benzerliğiyle birleştirmek zorunda.
3. Bağlam penceresi: 1024 token’a İngilizce yaklaşık 4.8 byte/token ile yaklaşık 4,900 byte, Türkçe 2.09 byte/token ile
yaklaşık 2,100 byte sığıyor. Aynı pencerede yarıdan az Türkçe metin var.
4. Hafta 8'le bağlama: Hafta 8'deki 160K parametrelik karakter modelini sadece Türkçe metinle
eğitmiştim ve ürettiğim metin Türkçe harflerle, Türkçe’ye benzer yapıda çıkıyordu.
GPT-2 124M ise 775 kat daha büyük ama Türkçeyi yazamıyor. Demek ki sorun model boyutu değil,
verinin dili ve tokenizer.
"""