# Görev 4 [Vikikaynak'tan yaklaşık 1 MB Türkçe metin derleme.]
# BU İLK DENEME

"""
MANTIK:
1. Vikikaynak'in "Eser" (mainspace) sayfalarinin listesini MediaWiki
   API'siyle cekiyoruz (action=query&list=allpages).
2. Her sayfanin duz metnini (action=parse&prop=wikitext, sonra basit
   bir wikitext temizligiyle) indirip biriktiriyoruz.
3. Toplam boyut yaklasik 1 MB'a ulasana kadar devam ediyoruz (Tiny Shakespeare
   ile benzer buyuklukte tutmak icin. (adil karsilastirma saglar)).
4. Sonucu turkce_metin.txt olarak kaydediyoruz. Bunu daha sonra
   train_turkish_gpt.py ile ayni klasore koyacagiz.

Kullanim:
    pip install requests
    python3 compile_turkish_corpus.py
"""

import re
import time
import requests

API_URL = "https://tr.wikisource.org/w/api.php"
TARGET_BYTES = 1_100_000  # Tiny Shakespeare'e yakın (1,115,394 karakter)
OUTPUT_PATH = "turkce_metin.txt"

def clean_wikitext(text):
    """
    Wikitext'teki markup'i (kategori etiketleri, referans/sablon
    kaliplari, dosya linkleri vs.) temizleyip duz metne cevirir.
    Mukemmel olmasi gerekmiyor amac, egitim verisini asiri markup
    gurultusunden arindirmak.
    """
    text = re.sub(r'\{\{[^{}]*\}\}', '', text)          # {{şablonlar}}
    text = re.sub(r'\[\[Kategori:[^\]]*\]\]', '', text)   # [[Kategori:...]]
    text = re.sub(r'\[\[Dosya:[^\]]*\]\]', '', text)      # [[Dosya:...]]
    text = re.sub(r'\[\[[^\|\]]*\|([^\]]*)\]\]', r'\1', text)  # [[link|görünen]] -> görünen
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)        # [[link]] -> link
    text = re.sub(r"'''?", '', text)                        # ''italik''/'''kalın'''
    text = re.sub(r'<[^>]+>', '', text)                      # <ref>...</ref> gibi html etiketleri
    text = re.sub(r'\n{3,}', '\n\n', text)                   # fazla bos satir
    return text.strip()

def get_page_titles(limit=500):
    """Vikikaynak'in ana ad alanindaki (mainspace, ns=0) sayfa basliklarini listeler."""
    titles = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": "max",
        "format": "json",
    }
    while len(titles) < limit:
        r = requests.get(API_URL, params=params, headers={"User-Agent": "YZ50-hafta8/1.0"})
        data = r.json()
        batch = [p["title"] for p in data.get("query", {}).get("allpages", [])]
        titles.extend(batch)
        cont = data.get("continue", {}).get("apcontinue")
        if not cont or not batch:
            break
        params["apcontinue"] = cont
        time.sleep(0.2)  # API'yi yormamak icin kucuk bir bekleme
    return titles

def get_page_text(title):
    """Tek bir sayfanin wikitext'ini ceker ve temizler."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }
    r = requests.get(API_URL, params=params, headers={"User-Agent": "YZ50-hafta8/1.0"})
    data = r.json()
    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    return clean_wikitext(wikitext)

if __name__ == "__main__":
    print("Sayfa basliklari cekiliyor")
    titles = get_page_titles(limit=2000)
    print(f"{len(titles)} sayfa basligi bulundu.")

    collected = []
    total_len = 0

    for title in titles:
        if total_len >= TARGET_BYTES:
            break
        try:
            text = get_page_text(title)
        except Exception as e:
            print(f"  atlandi ({title}): {e}")
            continue

        if len(text) < 200:  # cok kisa sayfalari (yonlendirme, index vb.) atla
            continue

        collected.append(text)
        total_len += len(text)
        print(f"  + {title}  ({len(text)} karakter, toplam {total_len})")
        time.sleep(0.2)

    full_text = "\n\n".join(collected)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nBitti. {OUTPUT_PATH} yazildi - {len(full_text)} karakter, "
          f"{len(set(full_text))} benzersiz karakter.")