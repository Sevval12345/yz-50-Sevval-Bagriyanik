# Görev 3: encode / decode.

"""
1) decode(encode(metin)) == metin  (Turkce karakterli ve emojili ornekle)
2) Gecersiz UTF-8 token dizileri: decode ne yapmali, neden?
3) Ters yon (encode(decode(ids)) == ids) her zaman dogru degil.

Gerekli: gorev 2'de egitilen turkce_bpe_512.model (hafta-9 klasorunde).
Calistirma (gorev-3 klasorunden):
    python encode_decode_demo.py
"""

import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))

from tokenizer import BasicTokenizer, goster

model_path = os.path.join(HERE, "..", "turkce_bpe_512.model")
if not os.path.exists(model_path):
    print("turkce_bpe_512.model yok. Once hafta-9 klasorunden: train_tokenizer.py dosyasını ekle knk!!!")
    sys.exit(1)

tok = BasicTokenizer()
tok.load(model_path)
print(f"Tokenizer yuklendi: vocab {len(tok.vocab)}, {len(tok.merges)} birlestirme\n")

# ------------------------------------------------------------------
print("=" * 70)
print("1) decode(encode(metin)) == metin")
print("=" * 70)
metin = "Ankara'dan İstanbul'a gittim 👋 Çok güzeldi, şöyle bir ığ ö ü ç!"
ids = tok.encode(metin)
print(f"metin   : {metin}")
print(f"karakter: {len(metin)}   byte: {len(metin.encode('utf-8'))}   token: {len(ids)}")
print(f"token id: {ids}")
print(f"token'lar: {[goster(tok.vocab[i]) for i in ids]}")
geri = tok.decode(ids)
print(f"decode  : {geri}")
print(f"AYNI MI : {geri == metin}")

print("\nDiger ornekler:")
ornekler = [
    "",                                   # bos metin
    "a",                                  # tek karakter
    "ı",                                  # tek Turkce harf
    "İ",                                  # noktali buyuk i
    "Merhaba dünya!\nİkinci satır.",      # satir sonu
    "你好 مرحبا Привет",                   # Cince, Arapca, Rusca
    "👨‍👩‍👧‍👦 🇹🇷 ❤️",                            # birlesik emoji (cok byte'li)
]
for o in ornekler:
    ok = tok.decode(tok.encode(o)) == o
    print(f"  {'OK ' if ok else 'HATA'}  {len(o.encode('utf-8')):>3} byte -> {len(tok.encode(o)):>3} token   {o!r}")

# ------------------------------------------------------------------
korpus = os.path.join(HERE, "..", "turkce_metin2.txt")
if os.path.exists(korpus):
    with open(korpus, "r", encoding="utf-8") as f:
        parca = f.read()[:100_000]
    t0 = time.time()
    k_ids = tok.encode(parca)
    dt = time.time() - t0
    ayni = tok.decode(k_ids) == parca
    n_byte = len(parca.encode("utf-8"))
    print(f"\nKorpusun ilk 100,000 karakteri: gidis-donus ayni mi?: {ayni}")
    print(f"  {n_byte:,} byte -> {len(k_ids):,} token  (byte/token {n_byte / len(k_ids):.3f})  encode suresi {dt:.1f} sn")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) GECERSIZ UTF-8: model her token dizisini uretebilir")
print("=" * 70)

def dene(ad, ids):
    b = b"".join(tok.vocab[i] for i in ids)
    try:
        b.decode("utf-8")
        strict = "gecerli"
    except UnicodeDecodeError as e:
        strict = f"UnicodeDecodeError: {e.reason}"
    print(f"  {ad:<34} byte'lar: {b.hex(' ').upper():<12}")
    print(f"      varsayilan decode : {strict}")
    print(f"      errors='replace'  : {tok.decode(ids)!r}")

dene("tek 0x80 (devam byte'i, baslangic yok)", [0x80])
dene("tek 0xC3 ('ç'nin ilk yarisi)", [0xC3])
dene("ters sirali 0xA7, 0xC3", [0xA7, 0xC3])
dene("dogru sirali 0xC3, 0xA7 ('ç')", [0xC3, 0xA7])
emoji_ids = tok.encode("👋")
print(f"\n  👋 emojisi = {len(emoji_ids)} token: {emoji_ids} (bu kucuk vocab'ta hic birlesmedi)")
dene("emojinin ilk 3 byte'i (yarida kesilmis)", emoji_ids[:3])
dene("emojinin tamami", emoji_ids)

random.seed(0)
n_deneme, n_gecersiz = 1000, 0
for _ in range(n_deneme):
    b = b"".join(tok.vocab[random.randrange(len(tok.vocab))] for _ in range(10))
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        n_gecersiz += 1
print(f"\n  Rastgele 10 token'lik {n_deneme} dizi: {n_gecersiz} tanesi gecerli UTF-8 DEGIL (%{n_gecersiz / n_deneme * 100:.1f})")
print("  Varsayilan decode bunlarin hepsinde cokerdi, errors='replace' hicbirinde cokmuyor.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) TERS YON: encode(decode(ids)) == ids ? (her zaman dogru degil)")
print("=" * 70)
ids2 = [0xC3, 0xA7]
metin2 = tok.decode(ids2)
yeni = tok.encode(metin2)
print(f"  ids            : {ids2}  -> decode -> {metin2!r}")
print(f"  encode(decode) : {yeni}   ayni mi: {yeni == ids2}")
print("  Ayni metin iki farkli token dizisiyle yazilabilir; encode her zaman")
print("  en cok birlestirilmis (kanonik) olani uretir.")

"""
Bazı token dizileri geçerli UTF-8 değil: decode bu durumda ne yapmalı, neden?

decode, gecersiz UTF-8 byte dizilerinde errors="replace" kullanmali:
gecersiz byte'larin yerine � (U+FFFD) isareti koymali, hata
vermemeli. Sebep: dil modeli her adimda vocab uzerinde bir
olasilik dagilimindan token secer ve hicbir sey sectigi token
dizisinin gecerli UTF-8 olmasini garanti etmez. Ornegin 0xC3 tek
basina 'ç'nin ilk yarisidir, ikinci yarisi gelmezse gecersizdir; bir
emojinin 4 byte'inin ilk 3'unde uretim kesilirse de ayni sey olur.
Python'un varsayilan decode'u bu durumda UnicodeDecodeError firlatir
ve uretim sirasinda program coker. replace ile program calismaya
devam eder, kullanici bozuk kismi � olarak gorur. Videoda anlatildigina
gore OpenAI da ayni ayari kullaniyor.
"""