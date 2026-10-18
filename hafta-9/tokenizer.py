# Görev 2: get_stats, merge, BasicTokenizer.train, save/load. (önce burası yazıldı)
# Görev 3: BasicTokenizer.encode ve decode eklendi. (görev 2 üstüne eklendi. yani bu py dosyası üzerinde çalıştım)
# Görev 4: GPT-4 regex deseni ve RegexTokenizer eklendi. (görev 2+3 üstüne eklendi. yani bu py dosyası üzerinde çalıştım)

from collections import Counter

def goster(tok_bytes):
    """Gecerli UTF-8 ise metin ('ı'), degilse hex (<C4>). Tek byte'lar harfin yarisi olabilir."""
    try:
        return repr(tok_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return "<" + tok_bytes.hex(" ").upper() + ">"

def get_stats(ids):
    """
    Ardisik ikililerin kac kez gectigini sayar.
        [1, 2, 3, 1, 2]  ->  {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, idx):
    """
    ids listesinde `pair` ikilisinin her gectigi yeri tek bir `idx` token'iyla
    degistirir.
        merge([1, 2, 3, 1, 2], (1, 2), 4)  ->  [4, 3, 4]
    """
    a, b = pair
    newids = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BasicTokenizer:
    def __init__(self):
        self.merges = {}      # (id1, id2) -> yeni id   (sırali: ekleme sirasi = birlestirme sirasi)
        self.vocab = self._build_vocab()
        self.history = []     # egitim gecmisi (her birlestirme icin bir kayit)

    def _build_vocab(self):
        """id -> bytes. 0-255 tek byte'lar, sonrasi birlesenlerin byte toplami."""
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        `text` uzerinde BPE egitir; vocab_size'a ulasana kadar en sik ikiliyi
        birlestirir. Gerekli birlestirme sayisi: vocab_size - 256.
        Dondurdugu: egitim metninin son token listesi.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))          # 0-255 arasi tamsayilar
        self.n_bytes = len(ids)
        self.merges = {}
        self.history = []

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)      # en sık ikili
            idx = 256 + i
            count = stats[pair]
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.history.append({
                "no": i + 1, "pair": pair, "id": idx,
                "count": count, "n_tokens": len(ids),
            })
            if verbose and (i + 1) % 50 == 0:
                print(f"  birlestirme {i + 1}/{num_merges}: to%ken sayisi {len(ids):,}")

        self.vocab = self._build_vocab()
        self.n_tokens = len(ids)
        return ids

    def decode(self, ids):
        """
        Token id listesi -> metin.
        Her id'nin byte karsiligini birlestirir, UTF-8 olarak yorumlar.
        Model HER token dizisini uretebilir, bunlarin hepsi gecerli UTF-8 degildir
        (orn. tek basina 0x80 ya da bir harfin yarisi). Python'un varsayilani
        UnicodeDecodeError firlatir ve program coker. errors="replace" ile gecersiz
        byte'lar yerine standart '\ufffd' isareti konur.
        """
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    def encode(self, text):
        """
        Metin -> token id listesi.
        1) UTF-8 byte'larina cevir. 2) merges'te bulunan ikililerden EN ERKEN
        ogrenileni (en kucuk yeni id) birlestir. 3) birlestirilecek ikili kalmayana
        kadar tekrarla. Sira onemli: sonraki birlestirmeler, onceki birlestirmelerin
        urettigi token'lara dayanir.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                      # birlestirilecek ikili kalmadi
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def token_text(self, idx):
        """Bir token'in metin karsiligi (gecersiz UTF-8 parcalari icin '\\ufffd' koyar)."""
        return self.vocab[idx].decode("utf-8", errors="replace")

    def compression_ratio(self):
        """byte sayisi / token sayisi (egitim metni icin)."""
        return self.n_bytes / self.n_tokens

    def save(self, path):
        """Her satirda bir birlestirme: 'id1 id2'. Satir sirasi = birlestirme sirasi."""
        with open(path, "w", encoding="utf-8") as f:
            for (p0, p1) in self.merges:
                f.write(f"{p0} {p1}\n")

    def load(self, path):
        merges = {}
        idx = 256
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()


# Görev 4: GPT-4 (cl100k_base) regex deseni ve RegexTokenizer

""" 
Metni BPE'den ONCE parcalara boler; BPE her parcanin icinde ayri calisir,
parcalar arasinda birlestirme olmaz. 7 alternatif:
  1) '(?i:[sdmt]|ll|ve|re)            Ingilizce kisaltmalar: 's 'd 'm 't 'll 've 're
  2) [^\r\n\p{L}\p{N}]?+\p{L}+          harf dizisi (kelime), onunde en fazla bir "harf/rakam/satir sonu
                                       olmayan" karakterle (cogu zaman bosluk): " gibi"
  3) \p{N}{1,3}                        rakamlar, en fazla 3'erli gruplar
  4)  ?[^\s\p{L}\p{N}]++[\r\n]*          noktalama / sembol dizisi
  5) \s*[\r\n]                        satir sonlari
  6) \s+(?!\S)                        sonrasinda kelime baslamayan bosluklar
  7) \s+                               kalan bosluklar
\p{L} = Unicode "harf" sinifi: c g i o s u (ve buyukleri) kelimeyi bolmez.
NOT: tiktoken'in guncel surumu ayni deseni sahiplenici nicelik belirteclerle (++)
yazar; 3000+ metinde ayni bolmeleri verdigini test ettik.
""" 
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    """
    BasicTokenizer + regex ile on bolme. Python'un `re` modulu Unicode harf sinifini
    (p{L}) bilmez, bu yuzden ucuncu parti `regex` paketi gerekir:
        python -m pip install regex
    """

    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        try:
            import regex
        except ImportError:
            raise ImportError("`regex` paketi gerekli: python -m pip install regex")
        self.pattern = pattern
        self.compiled = regex.compile(pattern)
        self.cache = {}

    def split(self, text):
        """Metni desene gore parcalara boler. ''.join(parcalar) == text."""
        return self.compiled.findall(text)

    def train(self, text, vocab_size, verbose=False):
        """
        Her parcanin ICINDE ayri BPE. Hiz icin ayni parcalari tek seferde sayariz:
        {parca: kac_kez_gecti}. Birlestirme her benzersiz parcada bir kez yapilir,
        istatistikler gecis sayisiyla agirliklanir. Sonuc, her gecisi tek tek
        islemekle ayni, ama cok daha hizli.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunk_counts = Counter(self.split(text))
        words = {c: list(c.encode("utf-8")) for c in chunk_counts}   # parca -> id listesi
        self.n_bytes = len(text.encode("utf-8"))
        self.n_chunks = sum(chunk_counts.values())
        self.n_unique_chunks = len(chunk_counts)
        self.merges = {}
        self.history = []
        self.cache = {}
        fixed_tokens = 0          # tek token'a inmis parcalarin toplam katkisi

        for i in range(num_merges):
            stats = {}
            for c, ids in words.items():
                k = chunk_counts[c]
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + k
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            count = stats[pair]

            for c in list(words):
                ids = words[c]
                new = merge(ids, pair, idx)
                if len(new) == 1:                      # parca tek token oldu, artik dokunulmaz
                    fixed_tokens += chunk_counts[c]
                    del words[c]
                else:
                    words[c] = new

            self.merges[pair] = idx
            n_tokens = fixed_tokens + sum(len(ids) * chunk_counts[c] for c, ids in words.items())
            self.history.append({"no": i + 1, "pair": pair, "id": idx,
                                 "count": count, "n_tokens": n_tokens})
            if verbose and (i + 1) % 50 == 0:
                print(f"  birlestirme {i + 1}/{num_merges}: token sayisi {n_tokens:,}")

        self.vocab = self._build_vocab()
        self.n_tokens = self.history[-1]["n_tokens"] if self.history else self.n_bytes

    def load(self, path):
        super().load(path)
        self.cache = {}

    def _encode_chunk(self, chunk):
        """Tek bir parcayi BPE ile encode eder (sik gecen parcalar onbellekte)."""
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        """Once regex ile bol, sonra her parcayi ayri encode et. decode BasicTokenizer'dan."""
        out = []
        for chunk in self.split(text):
            out.extend(self._encode_chunk(chunk))
        return out 