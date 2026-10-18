# Görev 4: GPT-4 regex deseni Türkçe'de ne yapıyor?

"""
1) Desenin tasarlandigi dil: Ingilizce kisaltmalar
2) Turkce harfler kelimeyi bolmuyor (\\p{L})
3) Kesme isaretli Turkce kelimeler: Ankara'dan, Istanbul'da ...
4) Hangi ekler boluyor, hangileri bolmuyor?
5) Duz (') ve kivrik (') kesme isareti
6) Egitilmis tokenizer'larla BPE token'lari (modeller varsa)
7) tiktoken'in GPT-4 deseniyle ayni mi? (tiktoken kuruluysa)
"""

import os
import sys
import regex

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))

from tokenizer import BasicTokenizer, RegexTokenizer, GPT4_SPLIT_PATTERN, goster

P = regex.compile(GPT4_SPLIT_PATTERN)
INGILIZCE_KISALTMA = regex.compile(r"(?i)^'(?:[sdmt]|ll|ve|re)$")

def bol(s):
    return P.findall(s)

# ------------------------------------------------------------------
print("=" * 70)
print("1) DESEN INGILIZCE ICIN YAZILMIS: kisaltmalari ayiriyor")
print("=" * 70)
for s in ["I'm don't we've They're it's", "he'll she'd"]:
    print(f"  {s!r:<34} -> {bol(s)}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) TURKCE HARFLER KELIMEYI BOLMUYOR (\\p{L} Unicode harf sinifi)")
print("=" * 70)
s = "Çok güzeldi, şöyle bir ığ çağdaş şoförün!"
print(f"  {s!r}")
print(f"  -> {bol(s)}")
print("  (Bosluk bir SONRAKI kelimeye yapisiyor; noktalama ayri parca.)")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) KESME ISARETLI TURKCE KELIMELER")
print("=" * 70)
kelimeler = ["Ankara'dan", "İstanbul'da", "İzmir'de", "Kadıköy'den", "Ankara'ta",
             "Ankara'sı", "Ankara'mız", "Türkiye'nin", "Ali'ye", "Ankara'ya",
             "Ahmet'in", "Ankara'lı"]
print(f"  {'kelime':<14}{'parcalar':<36}not")
print("  " + "-" * 66)
for w in kelimeler:
    parcalar = bol(w)
    ayrildi = [p for p in parcalar if INGILIZCE_KISALTMA.match(p)]
    not_ = f"<- {ayrildi[0]} Ingilizce kisaltma sanildi, ek bolundu" if ayrildi else ""
    print(f"  {w:<14}{str(parcalar):<36}{not_}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) HANGI EKLER BOLUNUYOR? (Ankara' + ek)")
print("=" * 70)
ekler = ["da", "de", "dan", "den", "dır", "dir", "ta", "te", "tan", "ten", "sı", "si", "sında",
         "mız", "miz", "ya", "ye", "a", "e", "yı", "yi", "nın", "nin", "ın", "in", "lı", "li", "la", "le", "ki"]
bolunen, bolunmeyen = [], []
for ek in ekler:
    parcalar = bol("Ankara'" + ek)
    (bolunen if any(INGILIZCE_KISALTMA.match(p) for p in parcalar) else bolunmeyen).append(ek)
print(f"  BOLUNEN    ({len(bolunen)}): {' '.join(chr(39) + e for e in bolunen)}")
print(f"  BOLUNMEYEN ({len(bolunmeyen)}): {' '.join(chr(39) + e for e in bolunmeyen)}")
print("  Ek s, d, m ya da t harfiyle basliyorsa (ya da ll/ve/re ile) Ingilizce")
print("  kisaltma kurali devreye giriyor ve eki ikiye boluyor.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("5) DUZ (') VE KIVRIK (’) KESME ISARETI")
print("=" * 70)
for s in ["Ankara'dan", "Ankara’dan"]:
    print(f"  {s!r:<14} (U+{ord(s[6]):04X}) -> {bol(s)}")
print("  Kisaltma kurali sadece duz ASCII kesme isaretini taniyor.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("6) BPE TOKEN'LARI (egitilmis tokenizer'larla)")
print("=" * 70)
yol_regex = os.path.join(HERE, "..", "turkce_bpe_regex_512.model")
yol_basic = os.path.join(HERE, "..", "turkce_bpe_512.model")
tokenizerlar = []
if os.path.exists(yol_basic):
    bt = BasicTokenizer(); bt.load(yol_basic); tokenizerlar.append(("regex YOK (gorev 2)", bt))
if os.path.exists(yol_regex):
    rt = RegexTokenizer(); rt.load(yol_regex); tokenizerlar.append(("regex VAR (gorev 4)", rt))
if not tokenizerlar:
    print("  model dosyasi yok; once train_tokenizer.py ve train_regex_tokenizer.py")
for ad, tok in tokenizerlar:
    print(f"\n  {ad}")
    for w in ["Ankara'dan", "İstanbul'da", "Ankara'ya"]:
        ids = tok.encode(w)
        print(f"    {w:<14} {len(ids):>2} token: {[goster(tok.vocab[i]) for i in ids]}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("7) tiktoken'in GPT-4 (cl100k_base) deseniyle ayni bolmeler mi?")
print("=" * 70)
try:
    import inspect
    import tiktoken_ext.openai_public as op
    src = inspect.getsource(op.cl100k_base)
    ns = {}
    # pat_str'yi fonksiyonun kaynagindan al (sozluk anahtari "pat_str")
    ilk = src.index('"pat_str": r"""') + len('"pat_str": r"""')
    pat_str = src[ilk:src.index('""",', ilk)]
    T = regex.compile(pat_str)
    denenen = kelimeler + ["I'm don't we've", "Çok güzeldi, şöyle bir ığ", "Merhaba   dünya!!\n\n  Nasılsın?  ", "12345678 1.000.000"]
    farkli = [x for x in denenen if T.findall(x) != P.findall(x)]
    print(f"  {len(denenen)} metin denendi, farkli bolunen: {len(farkli)}")
    print("  (tiktoken desenini sahiplenici nicelik belirteclerle ++ yazar; sonuc ayni olmali)")
except Exception as e:
    print(f"  tiktoken desenine erisilemedi ({e.__class__.__name__}); python -m pip install tiktoken")


"""
Desen Ingilizce icin yazilmis: 's 'd 'm 't 'll 've 're kisaltmalarini
kelimeden ayiriyor. Turkce'de kesme isareti kisaltma degil, ozel isimden eki
ayirir: Ankara'dan, Istanbul'da. Eki d, t, s ya da m ile basliyorsa desen
bunu Ingilizce kisaltma sanip ikiye boluyor: Ankara'dan -> Ankara, 'd, an.
Denedigim 30 ekin 15'i boluyor ('da 'de 'dan 'den 'dir 'ta 'ten 'si 'mız ...),
15'i bolunmuyor ('ya 'ye 'nin 'in 'lı 'le ...). Parcalar arasinda BPE
birlestirmesi yapilamadigi icin bolunen eklerin tek bir token olmasi
mumkun degil. Ayrica desen sadece duz apostrofu taniyor; kivrik apostrofla
(’) ayni kelime ['Ankara', '’dan'] diye farkli bolunuyor.
Turkce harfler (ç ğ ı ö ş ü) \p{L} sayesinde kelimeyi bolmuyor, yani desen
kelime sinirlarini Turkce'de de dogru buluyor; sorun sadece kesme isaretli
eklerde. Kendi 512 vocab'li tokenizer'imda bu bolmenin token sayisina etkisi
gorulmedi (Ankara'dan ve Ankara'ya ikisi de 8 token), cunku kucuk vocab'ta ek
token'lari zaten ogrenilmemis.

---------------------------------------------------------------------

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