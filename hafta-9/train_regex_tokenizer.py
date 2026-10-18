#Görev 4: GPT-4'un regex deseniyle ön bölme + BPE.

"""
Ayni korpusta, ayni vocab_size ile RegexTokenizer egitir ve gorev 2'deki
BasicTokenizer ile karsilastirir.
"""

import os
import sys
import time
import regex
from tokenizer import BasicTokenizer, RegexTokenizer, goster

HERE = os.path.dirname(os.path.abspath(__file__))
corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "turkce_metin2.txt")
vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
model_path = os.path.join(HERE, f"turkce_bpe_regex_{vocab_size}.model")
basic_path = os.path.join(HERE, f"turkce_bpe_{vocab_size}.model")

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Korpus: {corpus_path}")
print(f"karakter: {len(text):,}   byte: {len(text.encode('utf-8')):,}")
print(f"vocab_size: {vocab_size}  ->  {vocab_size - 256} birlestirme\n")

# ------------------------------------------------------------------
print("=" * 72)
print("1) METIN ONCE PARCALARA BOLUNUYOR")
print("=" * 72)
rt = RegexTokenizer()
ornek = text[:120]
print(f"ilk 120 karakter : {ornek!r}")
print(f"parcalar         : {rt.split(ornek)}")
print("  (Bosluk bir SONRAKI kelimeye yapisir: ' gibi'. BPE parcalar arasinda birlestirmez.)")

# ------------------------------------------------------------------
t0 = time.time()
rt.train(text, vocab_size, verbose=True)
print(f"\nparca sayisi: {rt.n_chunks:,}   benzersiz parca: {rt.n_unique_chunks:,}")
print(f"egitim suresi: {time.time() - t0:.1f} sn")

# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("2) ILK 20 BIRLESTIRME (regex'li)")
print("=" * 72)


def tur(tok_bytes):
    try:
        s = tok_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return "harfin yarisi (UTF-8 parcasi)"
    if len(s) == 1 and ord(s) > 127:
        return "byte'lardan tek harf"
    if s.startswith(" ") and len(s) > 1:
        return "bosluk + harfler"
    return ""


print(f"{'no':<4}{'sol':<12}{'sag':<12}{'-> yeni token':<16}{'gecis':>9}  not")
print("-" * 72)
for h in rt.history[:20]:
    a, b = h["pair"]
    yeni = rt.vocab[h["id"]]
    print(f"{h['no']:<4}{goster(rt.vocab[a]):<12}{goster(rt.vocab[b]):<12}"
          f"{goster(yeni):<16}{h['count']:>9,}  {tur(yeni)}")

# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("3) SIKISTIRMA ORANI (regex'li) = byte sayisi / token sayisi")
print("=" * 72)
n_bytes, n_chars = rt.n_bytes, len(text)
print(f"{'birlestirme':<13}{'vocab':<8}{'token sayisi':<16}{'byte/token':<12}{'karakter/token'}")
print("-" * 62)
for k in sorted({20, 50, 100, 200, len(rt.history)}):
    n_tok = rt.history[k - 1]["n_tokens"]
    print(f"{k:<13}{256 + k:<8}{n_tok:<16,}{n_bytes / n_tok:<12.3f}{n_chars / n_tok:.3f}")
print(f"\nNIHAI (regex'li): {n_bytes:,} byte -> {rt.n_tokens:,} token   oran = {n_bytes / rt.n_tokens:.3f}")

# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("4) GOREV 2 (regex'siz) ILE KARSILASTIRMA")
print("=" * 72)
if not os.path.exists(basic_path):
    print(f"{basic_path} yok. Karsilastirma icin once: python train_tokenizer.py")
else:
    bt = BasicTokenizer()
    bt.load(basic_path)
    t0 = time.time()
    basic_n_tokens = len(bt.encode(text))
    print(f"(regex'siz tokenizer ile korpus encode edildi: {time.time() - t0:.0f} sn)\n")

    gecen = regex.compile(r"\p{L}[^\p{L}\p{N}]")        # harften sonra harf/rakam olmayan bir karakter

    def ozet(tok):
        ogrenilen = [tok.vocab[i] for i in range(256, 256 + len(tok.merges))]
        metinler = [b.decode("utf-8", errors="replace") for b in ogrenilen]
        return {
            "kelime_sonu": [m for m in metinler if gecen.search(m)],
            "bosluk_basi": sum(1 for b in ogrenilen if b.startswith(b" ") and len(b) > 1),
            "ogrenilen": ogrenilen,
        }

    ob, orx = ozet(bt), ozet(rt)
    print(f"{'':<46}{'regex yok (gorev 2)':<22}{'regex var (gorev 4)'}")
    print("-" * 88)
    print(f"{'token sayisi':<46}{basic_n_tokens:<22,}{rt.n_tokens:,}")
    print(f"{'byte/token (sikistirma orani)':<46}{n_bytes / basic_n_tokens:<22.3f}{n_bytes / rt.n_tokens:.3f}")
    print(f"{'kelime sonu+bosluk/noktalama iceren token':<46}{len(ob['kelime_sonu']):<22}{len(orx['kelime_sonu'])}")
    print(f"{'bosluk ile baslayan token (+harf/isaret)':<46}{ob['bosluk_basi']:<22}{orx['bosluk_basi']}")
    print("  (Sayilar 256 ogrenilmis token icindeki adetler.)")

    ortak = set(ob["ogrenilen"]) & set(orx["ogrenilen"])
    print(f"\nIki tokenizer'in ogrendigi 256 token'in ortak olani: {len(ortak)}")
    for n in (20, 50, 100):
        ilk_b = set(ob["ogrenilen"][:n])
        ilk_r = set(orx["ogrenilen"][:n])
        print(f"  ilk {n:<3} birlestirmenin ortak olani: {len(ilk_b & ilk_r)}/{n}")

    sadece_b = [b for b in ob["ogrenilen"] if b not in orx["ogrenilen"]]
    sadece_r = [b for b in orx["ogrenilen"] if b not in ob["ogrenilen"]]
    print(f"\nSadece regex'siz tokenizer'da olan ilk 12 token:")
    print("  " + "  ".join(goster(b) for b in sadece_b[:12]))
    print(f"Sadece regex'li tokenizer'da olan ilk 12 token:")
    print("  " + "  ".join(goster(b) for b in sadece_r[:12]))
    print(f"\nregex'siz tokenizer'da KELIME SINIRINI ASAN ilk 12 token (kelime sonu + bosluk/noktalama):")
    print("  " + "  ".join(repr(m) for m in ob["kelime_sonu"][:12]))

rt.save(model_path)
print(f"\nEgitilmis regex'li tokenizer kaydedildi: {model_path}")

"""
Birleştirmeleri öncekiyle karşılaştırma:
- İlk 20 birleştirmenin 13'ü iki tokenizer'da ortak (ı ü ş ç ğ ar er in an ir ak al di).
- Regex'siz tokenizer'ın ilk 20'sinde kelime sınırını aşan 7 token vardı:
  'n ', 'e ', 'a ', 'i ', 'ı ', '. ', ', '. Regex'li tokenizer'da bunlar yok,
  yerlerine ' b', ' k', ' d', ' g', ' s' (boşluk + ilk harf) geldi.
- 256 öğrenilmiş token'dan kelime sonu+boşluk/noktalama içeren: 73 -> 0.
  Boşlukla başlayan token: 11 -> 92.
- Ortak token sayısı 116/256 (ilk 100 birleştirmede 60/100).
- Regex'siz tokenizer 'bir' ve 'bir ' diye aynı kelime için iki ayrı token
  öğrenmiş; regex'li tokenizer sadece ' bir' öğrenmiş.

Sıkıştırma oranı:
Regex'siz 968,125 / 456,822 = 2.119 byte/token, regex'li 968,125 / 462,812 =
2.092 byte/token. Regex parçalar arası birleştirmeyi yasakladığı için aynı
vocab boyutunda biraz daha az sıkıştırıyor (%1.3 fazla token), karşılığında
token'lar kelime sınırlarına uyuyor.
"""