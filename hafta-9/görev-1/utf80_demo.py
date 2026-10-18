# Görev 1: Metni byte'lara çevirme.

"""
Türkçe harflerin Unicode kod noktalarını ve UTF-8 byte'larını yazdırır,
hangi harfin kaç byte tuttuğunu gösterir, ve kod noktalarini doğrudan
vocab yapmanın neden kötü olduğunu sayılarla destekler.

Direkt python, ek paket gerekmiyo. Çalıştırmak için:
    python utf8_demo.py
"""

import os
import sys
import unicodedata

# ç ğ ı ö ş ü harflerinin hepsi bu cumlede var (Turkce pangram)
CUMLE = "Pijamalı hasta yağız şoföre çabucak güvendi."

def hex_bytes(ch):
    return " ".join(f"{b:02X}" for b in ch.encode("utf-8"))

def bin_bytes(ch):
    return " ".join(f"{b:08b}" for b in ch.encode("utf-8"))

# ------------------------------------------------------------------
print("=" * 66)
print(f"CUMLE: {CUMLE}")
print("=" * 66)

# 1) Cumledeki her benzersiz karakter: kod noktasi + UTF-8 byte'lari
print("\n1) HER KARAKTER: kod noktasi ve UTF-8 byte'lari (ilk gorulme sirasiyla)")
print(f"{'harf':<6}{'kod noktasi':<13}{'ondalik':<9}{'UTF-8 (hex)':<14}{'byte':<5}")
print("-" * 47)
gorulen = []
for ch in CUMLE:
    if ch in gorulen or ch == " ":
        continue
    gorulen.append(ch)
    print(f"{ch:<6}U+{ord(ch):04X}      {ord(ch):<9}{hex_bytes(ch):<14}{len(ch.encode('utf-8')):<5}")

# 2) Hangi harfler 1, hangileri 2 byte?
print("\n2) HANGI HARFLER KAC BYTE?")
kucuk = "abcçdefgğhıijklmnoöprsştuüvyz"
buyuk = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
sapkali = "âîûÂÎÛ"                      # Benim urettigim metin -> Omer Seyfettin metninde var (Kerâmet, Nâdan, Mâbed)
tum = kucuk + buyuk + sapkali
bir = [c for c in tum if len(c.encode("utf-8")) == 1]
iki = [c for c in tum if len(c.encode("utf-8")) == 2]
print(f"  1 byte : {' '.join(bir)}")
print(f"  2 byte : {' '.join(iki)}")
print(f"  -> Turkce alfabesindeki 12 harf (ç ğ ı ö ş ü + buyukleri) ve sapkali unluler 2 byte.")

# 3) UTF-8'in bit kalibi: 1, 2, 3 ve 4 byte'lik ornekler
print("\n3) UTF-8 BIT KALIBI")
for ch, aciklama in [("a", "ASCII"), ("ç", "Turkce"), ("ı", "Turkce (noktasiz i)"),
                     ("€", "Euro isareti"), ("👋", "emoji")]:
    print(f"  {ch}  U+{ord(ch):04X}  {len(ch.encode('utf-8'))} byte  {bin_bytes(ch)}   ({aciklama})")
print("  Baslangic byte'i kac byte oldugunu soyler (0, 110, 1110, 11110),")
print("  devam byte'lari hep '10' ile baslar.")

# 4) Butun cumle: karakter sayisi vs byte sayisi
print("\n4) BUTUN CUMLE")
b = CUMLE.encode("utf-8")
print(f"  karakter sayisi : {len(CUMLE)}")
print(f"  byte sayisi     : {len(b)}   (fark = 2 byte'lik harf sayisi: {len(b) - len(CUMLE)})")
print(f"  kod noktalari   : {[ord(c) for c in CUMLE][:12]} ...")
print(f"  UTF-8 byte'lari : {list(b)[:12]} ...")

# 5) UTF-8 / UTF-16 / UTF-32 karsilastirmasi
print("\n5) KODLAMA KARSILASTIRMASI (ayni cumle)")
for ad in ["utf-8", "utf-16-le", "utf-32-le"]:
    print(f"  {ad:<10}: {len(CUMLE.encode(ad))} byte")
print(f"  'a' harfi: UTF-8 {len('a'.encode('utf-8'))} byte, UTF-32 {len('a'.encode('utf-32-le'))} byte (3 tanesi sifir)")

# 6) Kod noktalarini dogrudan vocab yapsaydik
print("\n6) KOD NOKTALARINI DOGRUDAN VOCAB YAPSAYDIK")
toplam = sys.maxunicode + 1
kullanilan = sum(1 for i in range(toplam)
                 if unicodedata.category(chr(i)) not in ("Cn", "Cs", "Co"))
n_embd = 64
print(f"  Unicode kod noktasi alani : {toplam:,}")
print(f"  gercekten atanmis karakter: {kullanilan:,}   (Unicode {unicodedata.unidata_version})")
print(f"  embedding tablosu ({kullanilan:,} x {n_embd}): {kullanilan * n_embd:,} parametre")
print(f"  + cikis katmani (ayni boyut)             : {kullanilan * n_embd:,} parametre")
print(f"  hafta 8'deki butun modelimiz             : 164,839 parametre")
print(f"  UTF-8 byte vocab'i                       : 256 (sabit, hic degismez)")

# 7) Kendi Turkce korpusumda: karakter vs byte
korpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "turkce_metin2.txt")
if os.path.exists(korpus):
    with open(korpus, "r", encoding="utf-8") as f:
        metin = f.read()
    n_char, n_byte = len(metin), len(metin.encode("utf-8"))
    iki_byte = sum(1 for c in metin if len(c.encode("utf-8")) == 2)
    print("\n7) KENDI TURKCE KORPUSUN (turkce_metin2.txt)")
    print(f"  karakter: {n_char:,}   byte: {n_byte:,}   byte/karakter: {n_byte / n_char:.3f}")
    print(f"  2 byte'lik karakter: {iki_byte:,}  (%{iki_byte / n_char * 100:.1f})")
else:
    print("\n7) turkce_metin2.txt bulunamadi (hafta-9 klasorunden kopyala!!)")

"""
Kod noktalarını doğrudan vocab yapmak neden kötü?

1) Vocab çok büyük olur: Unicode'da 149,878 atanmış karakter var (Python'umdaki
   Unicode 15.1.0'a göre). n_embd=64 ile sadece embedding tablosu 9,592,192
   parametre, çıkış katmanıyla birlikte 19,184,384. Hafta 8'deki bütün modelim
   164,839 parametreydi, yani sadece vocab tabloları modelin yaklaşık 116 katı olurdu.
2) Satırların çoğu hiç kullanılmaz: Türkçe bir hikâye korpusunda Çince bir
   karakter hiç geçmez, o satır eğitimde güncellenmez.
3) Unicode standardı sürekli yeni karakter ekliyor (özellikle emoji), vocab'ın
   bir daha değişmeyeceğini garanti edemem.
UTF-8 byte'larıyla vocab 256 ve sabit, ama diziler uzuyor: 'ç' 2 token, emoji
4 token. BPE bu uzunluğu kısaltmak için var.

Hangi harfler kaç byte?

ç ğ ı ö ş ü (ve büyükleri, â î û) 2 byte, İngilizce
harfler 1 byte. 44 karakterlik cümlem 51 byte, fark cümledeki 7 iki byte'lık
harften geliyor. Korpusumda byte/karakter = 968,125 / 882,335 = 1.097.
"""