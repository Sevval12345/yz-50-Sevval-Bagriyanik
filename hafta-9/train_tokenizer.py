# GOREV 2 (hafta 9): BPE'yi kendi Turkce metninde egit.

import os
import sys
import time
from tokenizer import BasicTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "turkce_metin2.txt")
vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
model_path = os.path.join(HERE, f"turkce_bpe_{vocab_size}.model")

def goster(tok_bytes):
    """Gecerli UTF-8 ise metin ('ı'), degilse hex (<C4>). Tek byte'lar harfin yarisi olabilir."""
    try:
        return repr(tok_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return "<" + tok_bytes.hex(" ").upper() + ">"

def tur(tok_bytes):
    """Birlesen token'in ne oldugunu soyler (ASCII-disi tek harf / harfin yarisi / diger)."""
    try:
        s = tok_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return "harfin yarisi (UTF-8 parcasi)"
    if len(s) == 1 and ord(s) > 127:
        return "byte'lardan tek harf"
    return ""

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Korpus: {corpus_path}")
print(f"karakter: {len(text):,}   byte: {len(text.encode('utf-8')):,}")
print(f"vocab_size: {vocab_size}  ->  {vocab_size - 256} birlestirme yapilacak\n")

tok = BasicTokenizer()
t0 = time.time()
tok.train(text, vocab_size, verbose=True)
print(f"\negitim suresi: {time.time() - t0:.1f} sn")

# ---- ILK 20 BIRLESTIRME ----
print("\n" + "=" * 72)
print("ILK 20 BIRLESTIRME (metin olarak)")
print("=" * 72)
print(f"{'no':<4}{'sol':<12}{'sag':<12}{'-> yeni token':<16}{'gecis':>9}  not")
print("-" * 72)
n_harf = n_yarim = 0
for h in tok.history[:20]:
    a, b = h["pair"]
    yeni_bytes = tok.vocab[h["id"]]
    not_ = tur(yeni_bytes)
    n_harf += not_ == "byte'lardan tek harf"
    n_yarim += not_.startswith("harfin yarisi")
    print(f"{h['no']:<4}{goster(tok.vocab[a]):<12}{goster(tok.vocab[b]):<12}"
          f"{goster(yeni_bytes):<16}{h['count']:>9,}  {not_}")
print(f"\nilk 20 birlestirmenin {n_harf} tanesi byte'lari tek harfe ceviriyor, {n_yarim} tanesi bir harfin yarisini kuruyor.")

# ---- SIKISTIRMA ORANI ----
print("\n" + "=" * 72)
print("SIKISTIRMA ORANI = byte sayisi / token sayisi")
print("=" * 72)
print(f"{'birlestirme':<13}{'vocab':<8}{'token sayisi':<16}{'byte/token':<12}{'karakter/token'}")
print("-" * 62)
n_bytes, n_chars = tok.n_bytes, len(text)
print(f"{0:<13}{256:<8}{n_bytes:<16,}{1.0:<12.3f}{n_chars / n_bytes:.3f}")
kontrol = sorted({20, 50, 100, 200, len(tok.history)})
for k in kontrol:
    if k <= len(tok.history):
        n_tok = tok.history[k - 1]["n_tokens"]
        print(f"{k:<13}{256 + k:<8}{n_tok:<16,}{n_bytes / n_tok:<12.3f}{n_chars / n_tok:.3f}")
print(f"\nNIHAI: {n_bytes:,} byte -> {tok.n_tokens:,} token   sikistirma orani = {tok.compression_ratio():.3f}")

# ---- EN UZUN TOKEN'LAR ----
print("\n" + "=" * 72)
print("EN UZUN 15 TOKEN (egitimin sonunda ogrenilen 'kelime parcalari')")
print("=" * 72)
en_uzun = sorted(range(256, 256 + len(tok.history)), key=lambda i: -len(tok.vocab[i]))[:15]
for i in en_uzun:
    print(f"  {i:<5}{goster(tok.vocab[i]):<26}{len(tok.vocab[i])} byte")

tok.save(model_path)
print(f"\nEgitilmis tokenizer kaydedildi: {model_path}")

"""
- Hangi harf grupları önce birleşiyor?

İlk 20 birleştirmede üç tür grup var:

1) İki byte'lık Türkçe harfler: ı (36,413 geçiş -> en sık ikili), ü, ş, ç, ğ.
   Görev 1'de bu harflerin 2 byte tuttuğunu görmüştük. Tokenizer önce bu
   byte çiftlerini tek token olarak geri topluyor, çünkü her seferinde
   yan yana geliyorlar.

2) Harf + boşluk ve noktalama + boşluk: 'n ', 'e ', 'a ', 'i ', 'ı ', '. ', ', '.
   Yani kelime ve cümle sonları. Türkçe'de kelimeler ekle bittiği için
   kelime sonundaki harfler çok tekrar ediyor.

3) Sık harf çiftleri: ar, er, ir, in, an, ak, al, di. Bunların çoğu ek
   parçalarına benziyor (-lar/-ler, -mak, -dı gibi).

Sıkıştırma oranı:
968,125 byte -> 456,822 token, yani 968,125 / 456,822 = 2.119 byte/token
(token başına 1.931 karakter). İlk 20 birleştirme oranı 1.284'e, 100
birleştirme 1.738'e, 200 birleştirme 2.013'e çıkardı. Getiri azalıyor:
son 56 birleştirme token sayısını sadece %5 azalttı.
"""