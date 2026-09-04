"""
Turkce isim listesini hazirlama scripti.

Kaynak: tarikbahar/Turkish_names_and_surnames_dataset (GitHub)
Dosya: Turkce_isimler_listesi.txt

Bu script, indirilen ham dosyayi okuyup temizliyor ve names.txt ile
ayni formatta (her satirda bir isim) bir turkce_isimler.txt uretiyor.
"""

import unicodedata


def detect_encoding(path):
    """
    ADIM 1-2: Dosyanin encoding'ini tespit ediyoruz. Normal open(path, "r")
    ile acmaya calisirsak UTF-8 decode hatasi aliriz - dosya UTF-8 degil.
    chardet kutuphanesi, dosyanin byte'larina bakarak gercek encoding'i
    tahmin ediyor.
    """
    import chardet
    with open(path, "rb") as f:
        raw = f.read()
    result = chardet.detect(raw[:5000])
    print(f"Tespit edilen encoding: {result['encoding']} (guven: {result['confidence']})")
    return result['encoding']


def read_raw_lines(path, encoding):
    """
    Dosyayi tespit edilen encoding ile (bizim durumumuzda utf-16-le)
    okuyup satirlara boluyoruz.
    """
    with open(path, "r", encoding=encoding) as f:
        content = f.read()
    return content.splitlines()


def fix_turkish_lowercase(text):
    """
    ADIM 4: 'I harfi tuzagi'. Python'in standart .lower() metodu
    Turkce'ye ozel degil - 'I' harfini kucultmeye calisirken yanlis
    sonuc veriyor: 'İbrahim'.lower() -> 'i̇brahim' (i + gorunmeyen
    "combining dot" karakteri, iki ayri karakter). Bunu onlemek icin
    kucultmeden ONCE I/İ harflerini elle duz 'i'ye ceviriyoruz.
    """
    text = text.replace('İ', 'i').replace('I', 'i')
    return text.lower()


def clean_names(lines):
    """
    ADIM 3 + 5: Her satiri virgulden bolup sadece ismi aliyoruz
    (format: "Isim,cinsiyet_kodu" - k/e/u). Turkce kucuk harfe
    ceviriyoruz. Bosluklu (birden fazla kelimeli) isimleri ve
    tekrar eden kayitlari eliyoruz.
    """
    names = []
    seen = set()

    for line in lines:
        line = line.strip().lstrip('\ufeff')  # BOM karakteri varsa temizle
        if not line:
            continue

        parts = line.split(',')
        raw_name = parts[0].strip()

        name = fix_turkish_lowercase(raw_name)

        if ' ' in name or not name:
            continue  # birden fazla kelimeli kayitlari atla

        if name not in seen:
            seen.add(name)
            names.append(name)

    return names


def verify_character_set(names):
    """
    ADIM 6: Temizligin dogru calistigini kontrol etmek icin, kullanilan
    tum karakterleri alfabetik siraya dizip yazdiriyoruz. Beklenmedik
    bir karakter (ornegin combining dot gibi gorunmez bir sey) varsa
    burada fark ederiz.
    """
    chars = sorted(set(''.join(names)))
    print(f"Kullanilan karakterler ({len(chars)} adet): {chars}")

    # Her karakterin gorunmeyen/birlesik bir karakter olup olmadigini
    # kontrol edelim (unicodedata.category ile)
    for ch in chars:
        category = unicodedata.category(ch)
        if category.startswith('M'):  # M* = "Mark" kategorisi, birlesik karakterler
            print(f"  UYARI: '{ch}' birlesik/gorunmez bir karakter olabilir (kategori: {category})")


def write_clean_names(names, output_path):
    """
    Temizlenmis isim listesini, names.txt ile ayni formatta
    (her satirda bir isim) bir dosyaya yaziyoruz.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + '\n')
    print(f"'{output_path}' yazildi. Toplam {len(names)} isim.")


if __name__ == "__main__":
    raw_path = "Turkce_isimler_listesi.txt"
    output_path = "turkce_isimler.txt"

    # 1-2: Encoding tespit et ve dosyayi oku
    encoding = detect_encoding(raw_path)
    lines = read_raw_lines(raw_path, encoding)
    print(f"Ham dosyada toplam satir: {len(lines)}")
    print(f"Ilk 5 ham satir: {lines[:5]}")
    print()

    # 3-5: Temizle (virgulden bol, Turkce kucuk harfe cevir, gecersizleri ele)
    names = clean_names(lines)
    print(f"Temizlik sonrasi benzersiz isim sayisi: {len(names)}")
    print(f"Ilk 10 isim: {names[:10]}")
    print()

    # 6: Karakter setini dogrula
    verify_character_set(names)
    print()

    # Dosyaya yaz
    write_clean_names(names, output_path)