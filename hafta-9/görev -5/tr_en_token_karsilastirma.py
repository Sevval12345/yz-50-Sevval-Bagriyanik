# Görev 5: Türkçe ile İngilizceyi karşılaştır.

"""
Ayni paragrafin Turkcesi ve Ingilizcesi:
  1) GPT-4'un tokenizer'i (tiktoken, cl100k_base) ile token'lanir,
     kelime basina token sayisi hesaplanir.
  2) Ayni paragraflar kendi tokenizer'larinla (gorev 2 ve gorev 4) token'lanir.
  3) (Istege bagli) Kendi korpusundan 100,000 karakterlik bir parcada GPT-4 ile
     kendi tokenizer'larinin sikistirma orani karsilastirilir.
"""

import os
import sys
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))

from tokenizer import BasicTokenizer, RegexTokenizer, goster

# ---- Karsilastirilacak paragraf (bunu ai'a yazdırdım)----
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

# Turkce kelime (ekli) <-> Ingilizce karsiligi
ORNEKLER = [
    ("evlerimizden", "from our houses"),
    ("öğrencilerinizden", "from your students"),
    ("göremeyeceksiniz", "you will not be able to see"),
    ("gidebilirdik", "we could have gone"),
    ("okuyamadıklarımız", "the ones we could not read"),
]

enc = tiktoken.get_encoding("cl100k_base")

def say(text):
    """Metnin istatistikleri (cl100k_base ile)."""
    ids = enc.encode(text)
    n_word = len(text.split())
    n_byte = len(text.encode("utf-8"))
    return {"kelime": n_word, "karakter": len(text), "byte": n_byte,
            "token": len(ids), "ids": ids}

# ------------------------------------------------------------------
print("=" * 78)
print("1) GPT-4 (cl100k_base): AYNI PARAGRAFIN TURKCESI VE INGILIZCESI")
print("=" * 78)
print(f"EN: {EN}\n")
print(f"TR: {TR}\n")

en, tr = say(EN), say(TR)
print(f"{'':<12}{'kelime':>8}{'karakter':>10}{'byte':>7}{'token':>7}{'token/kelime':>14}{'byte/token':>12}")
print("-" * 70)
for ad, s in [("Ingilizce", en), ("Turkce", tr)]:
    print(f"{ad:<12}{s['kelime']:>8}{s['karakter']:>10}{s['byte']:>7}{s['token']:>7}"
          f"{s['token'] / s['kelime']:>14.2f}{s['byte'] / s['token']:>12.2f}")
print("-" * 70)
print(f"{'TR / EN':<12}{tr['kelime'] / en['kelime']:>8.2f}{tr['karakter'] / en['karakter']:>10.2f}"
      f"{tr['byte'] / en['byte']:>7.2f}{tr['token'] / en['token']:>7.2f}"
      f"{(tr['token'] / tr['kelime']) / (en['token'] / en['kelime']):>14.2f}")
print("\n  Turkce'de kelime sayisi daha az (eklemeli yapi), ama toplam token sayisi ve")
print("  token/kelime orani ne yonde? Yukaridaki 'TR / EN' satirina bak.")

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("2) TURKCE PARAGRAFIN ILK 30 TOKEN'I (cl100k_base)")
print("=" * 78)
parcalar = [goster(enc.decode_single_token_bytes(i)) for i in tr["ids"][:30]]
print("  " + "  ".join(parcalar))
print("\n   EN paragrafin ilk 30 token'i:")
print("  " + "  ".join(goster(enc.decode_single_token_bytes(i)) for i in en["ids"][:30]))

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("3) EKLEMELI YAPI: TEK TURKCE KELIME vs INGILIZCE KARSILIGI (cl100k_base)")
print("=" * 78)
print(f"{'Turkce':<20}{'tok':>4}  {'Ingilizce':<30}{'tok':>4}  {'TR/EN token':>12}")
print("-" * 76)
for t, e in ORNEKLER:
    nt, ne = len(enc.encode(t)), len(enc.encode(e))
    print(f"{t:<20}{nt:>4}  {e:<30}{ne:>4}  {nt / ne:>12.2f}")
print("\n  Turkce kelimelerin parcalanisi:")
for t, _ in ORNEKLER:
    print(f"    {t:<20} -> {[goster(enc.decode_single_token_bytes(i)) for i in enc.encode(t)]}")

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("4) KENDI TOKENIZER'LARIN (Turkce korpusta egitilmis, vocab 512)")
print("=" * 78)
yol_basic = os.path.join(HERE, "..", "turkce_bpe_512.model")
yol_regex = os.path.join(HERE, "..", "turkce_bpe_regex_512.model")
benim = []
if os.path.exists(yol_basic):
    t = BasicTokenizer(); t.load(yol_basic); benim.append(("kendi (regex yok, g.2)", t))
if os.path.exists(yol_regex):
    t = RegexTokenizer(); t.load(yol_regex); benim.append(("kendi (regex var, g.4)", t))

if not benim:
    print("  model dosyasi yok; once train_tokenizer.py ve train_regex_tokenizer.py")
else:
    print(f"{'tokenizer':<26}{'metin':<11}{'token':>7}{'token/kelime':>14}{'byte/token':>12}")
    print("-" * 70)
    for ad_metin, s in [("Turkce", tr), ("Ingilizce", en)]:
        print(f"{'GPT-4 cl100k_base':<26}{ad_metin:<11}{s['token']:>7}{s['token'] / s['kelime']:>14.2f}{s['byte'] / s['token']:>12.2f}")
        text = TR if ad_metin == "Turkce" else EN
        for ad, tok in benim:
            n = len(tok.encode(text))
            print(f"{ad:<26}{ad_metin:<11}{n:>7}{n / s['kelime']:>14.2f}{s['byte'] / n:>12.2f}")
        print()

    print("Turkce paragraf, kendi regex'siz tokenizer'inla (ilk 30 token):")
    tok = benim[0][1]
    print("  " + "  ".join(goster(tok.vocab[i]) for i in tok.encode(TR)[:30]))

# ------------------------------------------------------------------
korpus = os.path.join(HERE, "..", "turkce_metin2.txt")
if os.path.exists(korpus):
    print("\n" + "=" * 78)
    print("5) KENDI KORPUSUNDAN 100,000 KARAKTER: GPT-4 vs KENDI TOKENIZER'IN")
    print("=" * 78)
    with open(korpus, "r", encoding="utf-8") as f:
        parca = f.read()[:100_000]
    n_byte = len(parca.encode("utf-8"))
    n_gpt = len(enc.encode(parca))
    print(f"  {'tokenizer':<26}{'vocab':>8}{'token':>9}{'byte/token':>12}")
    print(f"  {'GPT-4 cl100k_base':<26}{enc.n_vocab:>8,}{n_gpt:>9,}{n_byte / n_gpt:>12.3f}")
    for ad, tok in benim:
        n = len(tok.encode(parca))
        print(f"  {ad:<26}{len(tok.vocab):>8,}{n:>9,}{n_byte / n:>12.3f}")
    print("  (Kendi tokenizer'lar bu parcayi egitimde gordu. cl100k_base'in egitim verisi bilinmiyor,")
    print("   bu yuzden bu satirlar adil bir yaris degil, sadece buyuklukleri gosteriyor: vocab 512 vs ~100 bin.)")

"""
GPT-4 (cl100k_base) ile aynı paragraf: İngilizce 68 kelime / 75 token (1.10 token/kelime),
Türkçe 43 kelime / 133 token (3.09 token/kelime). Türkçe %37 daha az kelimeyle yazılıyor
ama %77 daha fazla token tutuyor (133/75 = 1.77). Byte/token 4.85'ten 2.86'ya düşüyor.

Eklemeli yapının etkisi iki zıt yönde:
1) Kelime sayısını azaltıyor: ekler kelimeye yapıştığı için aynı içerik 43 kelimeye sığıyor.
2) Ama her kelime çok daha fazla token'a bölünüyor: 2.80 kat. İkincisi baskın çıktı.
Eklemeli kelimeler ek parçalarına bölünüyor: evlerimizden -> ev|ler|im|iz|den (5 token).
Tek bir kelime birkaç İngilizce kelimeye denk geliyor ama token maliyeti de ona yakın:
göremeyeceksiniz 7 token, 'you will not be able to see' de 7 token.
Paragrafta oran daha yüksek (1.77), çünkü kökler de parçalanıyor ve ş, ü, ö, ğ gibi
2 byte'lık harfler çoğu zaman tek başına token oluyor (öğrencilerinizden -> ö|ğ|renc|iler|...).

Kendi tokenizer'larımla (vocab 512, sadece Türkçe korpusta eğitilmiş): Türkçe paragraf 172
token (GPT-4: 133), İngilizce 264 token (GPT-4: 75). Tokenizer neyle eğitildiyse onu iyi
sıkıştırıyor: GPT-4'te Türkçe İngilizce'den 1.77 kat pahalı, benimkinde İngilizce Türkçe'den
1.53 kat pahalı. Kendi tokenizer'ım ekleri daha iyi hizalıyor (arkadaşlarımla ->
ar|kad|aş|lar|ım|la), ama vocab 200 kat küçük olduğu için toplamda daha çok token üretiyor.
"""