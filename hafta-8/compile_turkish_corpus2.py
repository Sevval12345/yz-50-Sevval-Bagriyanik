# Görev 4
# İkinci deneme

"""
Cikti: turkce_metin2.txt

v1'de iki sorun cikti:
  1) Sayfalar baslik sirasina gore cekildi ve korpusun %99'u iki buyuk
     iddianame sayfasindan geldi (tek alan, tek belge turu).
  2) Yuzlerce istek "Expecting value" hatasiyla atlandi: API JSON yerine
     bir hata sayfasi dondurdu (buyuk olasilikla cok hizli istek yuzunden).

v2'de:
  - Korpus YAZAR bazli kurulur. Her yazara bir karakter butcesi verirsin
    (YAZARLAR sozlugu), script o yazarin eserlerini aratip butce dolana
    kadar toplar. Hicbir yazar korpusa hakim olamaz.
  - Her eser ayrica MAX_PER_WORK ile sinirli.
  - Basarisiz istek 4 kereye kadar bekleyerek yeniden denenir.
  - Hukuki/siyasi belgeler (iddianame, kanun, bildiri...) basliga gore elenir.

KULLANIM:
    python compile_turkish_corpus2.py --ara "Ömer Seyfettin"      # ham arama sonuclari
    python compile_turkish_corpus2.py --onizle "Ömer Seyfettin"   # script neleri secer/eler, toplam boyut
    # uygun yazarlari asagidaki YAZARLAR sozlugune yaz, sonra:
    python compile_turkish_corpus2.py                              # derle (~10 dk surer)
"""

import html
import re
import sys
import time
import requests

API_URL = "https://tr.wikisource.org/w/api.php"
# Wikimedia kurali: User-Agent'ta iletisim bilgisi olsun.
HEADERS = {"User-Agent": "YZ50-hafta8/2.0 (egitim projesi; sevvalbgrynk@hotmail.com)"}

TARGET_CHARS = 1_100_000      # Tiny Shakespeare ~1.1M karakter
MAX_PER_WORK = 80_000         # tek bir eser en fazla bu kadar katki yapar
MIN_PAGE_CHARS = 200          # bundan kisa sayfalar (yonlendirme vb.) atlanir
MIN_WORK_BYTES = 2000         # bundan kucuk arama sonuclari (taslak/indeks) atlanir
OUTPUT_PATH = "turkce_metin2.txt"

# yazar -> bu yazardan en fazla kac karakter alinacak.
# Once --onizle ile yazarin Vikikaynak'ta yeterli eseri var mi bak.
# Onizleme sonucu: Vikikaynak'ta yeterli edebi metni olan tek yazar Ömer
# Seyfettin (77 hikaye, ~1 MB). Tiny Shakespeare de tek yazarli bir korpus.
YAZARLAR = {"Ömer Seyfettin": 1_100_000}

# Istersen tek tek eser basligi da ekleyebilirsin (yazar kotasi disinda).
ESERLER = []

# Hukuki / siyasi belgeleri basliga gore eleme. Karsilastirma icin Turkce
# harfler ASCII'ye indirgenir (Python'da "İ".lower() = "i" + gorunmez nokta
# oldugu icin dogrudan lower() ile eslestirme calismaz).
# Aramada yazarin adi gecmedigi halde sizan, bilinen yabanci sayfalar:
HARIC_BASLIK = ("Attilâ İlhan", "Ali Kemal Bey'e Cevabım", "Milli hükümet programı",
                "Erken Soğuk Savaş", "Marifetname", "Tarih II")

MAX_WORK_BYTES = 100_000      # tek sayfalik arama sonucu bundan buyukse (rapor, savunma...) atlanir

_FOLD = str.maketrans("İIıiŞşÇçĞğÖöÜüÂâÎîÛû", "iiiissccggoouuaaiiuu")

def fold(s):
    return s.translate(_FOLD).lower() 

# Kelime BASLANGICI eslesirse elenir (iddianame -> iddianamesi, iddianameler)
ATLA_KOK = ("iddianame", "kanun", "anayasa", "sozlesme", "antlasma", "bildiri",
            "tutanak", "teblig", "nutuk", "rapor", "konusma", "savunma", "cagri",
            "komisyon", "miting", "genelge", "yonetmelik", "protokol", "dilekce",
            "onerge", "referandum", "tbmm", "savci", "mahkeme", "muhtira")
# TAM kelime eslesirse elenir ("yasa" ama "yasam"/"yasak" degil)
ATLA_TAM = {"yasa", "yasasi", "karar", "karari", "kurul", "mesaj", "mesaji",
            "parti", "partisi", "meclis"}

def elenme_nedeni(title, size):
    """Bir arama sonucunu neden eledigimizi dondurur, elenmiyorsa None."""
    root = title.split("/")[0]
    if any(fold(root).startswith(fold(h)) for h in HARIC_BASLIK):
        return "elle haric tutuldu"
    words = re.findall(r"\w+", fold(root))
    if any(w.startswith(k) for w in words for k in ATLA_KOK) or any(w in ATLA_TAM for w in words):
        return "hukuki/siyasi baslik"
    if "/" not in title and size < MIN_WORK_BYTES:
        return f"cok kucuk (<{MIN_WORK_BYTES} bayt)"
    if "/" not in title and size > MAX_WORK_BYTES:
        return f"cok buyuk (>{MAX_WORK_BYTES} bayt)"
    return None

# Tipografik karakterleri sadelestir: ayni karakterin 3-4 farkli
# yazilisi vocab_size'i gereksiz sisiriyor.
NORMALIZE = {
    "’": "'", "‘": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "…": "...", "\u200e": "", "\u200f": "",
    "\xa0": " ", "\t": " ",
}

def api_get(params, retries=4):
    """Rate limit / gecici hata icin bekleyerek yeniden dener, JSON dondurur."""
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
            print(f"    HTTP {r.status_code}, {wait} sn bekleniyor...")
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** (attempt + 1)
            print(f"    hata ({e.__class__.__name__}), {wait} sn bekleniyor...")
        time.sleep(wait)
    return None

def clean_wikitext(text):
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:Kategori|Dosya|Category|File):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)     # dipnotlari metniyle birlikte sil !!
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)                                         # &mdash; &nbsp; &quot; &hellip; -> gercek karakter !!
    text = re.sub(r"^[:*#;]+[ ]*", "", text, flags=re.M)               # satir basi wiki girinti/liste isaretleri (":" "*" "#")
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)   # == Baslik == -> Baslik
    text = re.sub(r"^[\s\W_]+$", "", text, flags=re.M)                 # sadece sembol iceren satirlar
    for k, v in NORMALIZE.items():
        text = text.replace(k, v)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"^[ ]+", "", text, flags=re.M)                      # satir basi bosluklari
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def search_titles(query, limit=50):
    """--ara: ham arama sonuclari."""
    data = api_get({"action": "query", "list": "search", "srsearch": query,
                    "srnamespace": 0, "srlimit": limit, "srprop": "size", "format": "json"})
    if not data:
        print("Arama basarisiz.")
        return
    print(f"'{query}' icin sonuclar (baslik | bayt):")
    for h in data.get("query", {}).get("search", []):
        print(f"  {h['title']}  |  {h.get('size', '?')}")

def search_works(author, max_hits=150, elenenler=None):
    """
    Bir yazarin eserlerini aratir, (eser_basligi, bayt) listesi dondurur.
    - Alt sayfa sonuclari ("Roman/1. Bolum") ana esere ("Roman") indirgenir,
      boylece ayni eser iki kez gelmez.
    - elenme_nedeni() ile hukuki/siyasi basliklar ve cok kucuk/buyuk
      sayfalar elenir. Elenenler `elenenler` listesine (baslik, bayt, neden)
      olarak eklenir, --onizle bunlari gosterir.
    """
    works, seen = [], set()
    for offset in range(0, max_hits, 50):
        data = api_get({"action": "query", "list": "search", "srsearch": author,
                        "srnamespace": 0, "srlimit": 50, "sroffset": offset,
                        "srprop": "size", "format": "json"})
        hits = (data or {}).get("query", {}).get("search", [])
        if not hits:
            break
        for h in hits:
            title, size = h["title"], h.get("size", 0)
            root = title.split("/")[0]
            neden = elenme_nedeni(title, size)
            if neden:
                if elenenler is not None:
                    elenenler.append((title, size, neden))
                continue
            if root in seen:
                continue
            seen.add(root)
            works.append((root, size))
        time.sleep(1)
    return works

def pages_of_work(root):
    """Eserin kendisi + 'root/...' alt sayfalari."""
    titles = [root]
    params = {"action": "query", "list": "allpages", "apnamespace": 0,
              "apprefix": root + "/", "aplimit": "max", "format": "json"}
    while True:
        data = api_get(params)
        time.sleep(1)
        if not data:
            break
        titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
        cont = data.get("continue", {}).get("apcontinue")
        if not cont:
            break
        params["apcontinue"] = cont
    return titles

def page_text(title):
    data = api_get({"action": "parse", "page": title, "prop": "wikitext", "format": "json"})
    if not data:
        return ""
    return clean_wikitext(data.get("parse", {}).get("wikitext", {}).get("*", ""))

def collect_work(work, remaining):
    """Bir eserin sayfalarini ceker; en fazla min(MAX_PER_WORK, remaining) karakter."""
    texts, got = [], 0
    limit = min(MAX_PER_WORK, remaining)
    for title in pages_of_work(work):
        if got >= limit:
            break
        text = page_text(title)
        time.sleep(1)                           # API'yi yormamak icin
        if len(text) < MIN_PAGE_CHARS:
            continue
        text = text[: limit - got]              # kotayi asmasin
        texts.append(text)
        got += len(text)
        print(f"  + {title} ({len(text)} kr, eser toplami {got})")
    return texts, got

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--ara":
        search_titles(" ".join(sys.argv[2:]))
        sys.exit(0)

    if len(sys.argv) >= 3 and sys.argv[1] == "--onizle":
        author = " ".join(sys.argv[2:])
        elenenler = []
        works = search_works(author, elenenler=elenenler)
        total = sum(sz for _, sz in works)
        print("SECILENLER (baslik, bayt):")
        for title, size in works:
            print(f"  {size:>7}  {title}")
        if elenenler:
            print("\nELENENLER:")
            for title, size, neden in elenenler:
                print(f"  {size:>7}  {title}   [{neden}]")
        print(f"\n{len(works)} eser secildi, {len(elenenler)} sonuc elendi, toplam ~{total} bayt "
              f"(karakter sayisi biraz daha az olur)")
        sys.exit(0)

    groups = [(a, None, b) for a, b in YAZARLAR.items()]
    if ESERLER:
        groups.append(("(elle secilenler)", ESERLER, TARGET_CHARS))
    if not groups:
        print("YAZARLAR ve ESERLER bos. Once: python compile_turkish_corpus2.py --onizle \"yazar adi\"")
        sys.exit(1)

    chunks, report, total = [], [], 0
    for label, works, budget in groups:
        if total >= TARGET_CHARS:
            break
        print(f"\n=== {label} (butce {budget}) ===")
        if works is None:
            works = [t for t, _ in search_works(label)]
            print(f"  {len(works)} eser bulundu")
        got_group = 0
        for work in works:
            if got_group >= budget or total >= TARGET_CHARS:
                break
            texts, got = collect_work(work, min(budget - got_group, TARGET_CHARS - total))
            chunks += texts
            got_group += got
            total += got
        report.append((label, got_group))

    full = "\n\n".join(chunks)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full)

    print(f"\n{OUTPUT_PATH}: {len(full)} karakter, {len(set(full))} benzersiz karakter")
    print("\nYAZAR BAZLI PAY:")
    for label, n in report:
        pay = n / max(len(full), 1) * 100
        uyari = "   <-- cok yuksek" if (pay > 35 and len(report) > 1) else ""
        print(f"  %{pay:5.1f}  {n:>7}  {label}{uyari}")