import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """
    Görev 3: Tek bir self-attention head.

    key, query, value: ucu de x'ten nn.Linear (bias'siz) ile uretilen
    kucuk (head_size boyutunda) projeksiyonlar.
    - query: "ben ne ariyorum"
    - key:   "ben ne sunuyorum"
    - value: "beni secersen sana verecegim bilgi" (x'in kendisi degil,
    ayri bir ogrenilmis projeksiyon)

    wei = q @ k.T: her token ciftinin ilgi puani (B,T,T). head_size'in
    karekoku ile olcekliyoruz (asagidaki demo'da neden gerekli oldugunu
    sayiyla gosteriyoruz). tril ile GELECEK pozisyonlari -inf yapip
    softmax'a veriyoruz - boylece bir token sadece KENDISINDEN ONCEKI
    (ve kendisi dahil) tokenlara bakabiliyor, gelecegi goremiyor.

    forward() ciktisinin yaninda self.wei'yi de SAKLIYORUZ (hafta 4-6'daki
    kendi katmanlarimizin self.out'u sakladigi gibi) - boylece disaridan
    "bu token hangilerine ne kadar bakti" diye inceleyebiliyoruz.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0, causal=True):
        super().__init__()
        self.head_size = head_size
        self.causal = causal  # GOREV 5b: False yapilirsa tril maskesi uygulanmaz
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer: ogrenilmeyen ama modelle birlikte tasinan sabit.
        # Gecen haftalarin running_mean/running_var'iyla ayni mantik.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5   # (B, T, T)
        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # self.causal=False ise: GOREV 5b - maske YOK, her token GELECEKTEKI
        # tokenlara da bakabiliyor (kendi hedefi de dahil - veri sizintisi).
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        self.wei = wei  # incelemek icin disariya acik tutuyoruz

        v = self.value(x)    # (B, T, head_size)
        out = wei @ v         # (B, T, head_size)
        return out

class BigramLanguageModel(nn.Module):
    """
    Görev 1: Hafta 3'teki bigram modelinin (27x27 agirlik matrisi, kendi
    elimizle yazdigimiz) torch.nn diliyle yazilmis hali. Artik kendi
    Linear/Embedding siniflarimizi degil, GERCEK torch.nn.Module'u
    kullaniyoruz.

    Tek katman: nn.Embedding(vocab_size, vocab_size). Bu, her karakter
    icin DOGRUDAN bir sonraki karakterin logit'lerini tutan bir tablo -
    yani token_embedding_table[i] = i'den sonra hangi karakterin
    gelecegine dair (henuz egitilmemis) tahmin. Baglam onemli degil,
    sadece SON karaktere bakiyor (bigram = 2'li, bir onceki + simdiki).

    Bu, ileride attention ekleyecegimiz modelin "karsilastirma tabani".
    attention'in gercekten fayda sagladigini gormek icin once
    attention'siz halin val loss'unu bilmemiz gerekiyor.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) - B ornek, T uzunlugunda baglam (tam sayi indeksleri)
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Metin uretmek icin: mevcut baglamdan (idx) bir sonraki karakterin
        olasilik dagilimini al, ORNEKLEME yap (argmax degil, cesitlilik
        icin), uretilen karakteri baglama ekle, tekrarla. Bigram modelde
        baglamin uzunlugu onemli degil (sadece son karaktere bakiyor),
        ama fonksiyonu genel yaziyoruz ki ileride attention'li modelde de
        aynen kullanabilelim.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # sadece son zaman adiminin logit'leri: (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class AttentionLanguageModel(nn.Module):
    """
    Görev 4: Head'i (gorev 3) bigram modelinin (gorev 1) icine koyuyoruz.

    Eski BigramLanguageModel: tek katman, dogrudan vocab_size'a cikis
    yapan nn.Embedding -> aradan hicbir sey gecmiyordu.

    Bu model:
      1. token_embedding_table: karakteri n_embd boyutlu bir vektore cevir
      2. position_embedding_table: pozisyon bilgisini ekle (gorev 3)
      3. sa_head: self-attention ile gecmise bakarak vektorleri harmanla
      4. lm_head: harmanlanmis vektoru tekrar vocab_size boyutuna (logit)
         cevirmek icin SON bir nn.Linear katmani

    Block_size, hem position_embedding_table'in boyutunu hem de Head
    icindeki tril maskesinin boyutunu belirliyor - bu yuzden model
    ayni anda en fazla block_size kadar geriye bakabiliyor.
    """

    def __init__(self, vocab_size, n_embd, block_size, head_size=None, dropout=0.0, causal=True):
        super().__init__()
        head_size = head_size if head_size is not None else n_embd
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, head_size, block_size, dropout=dropout, causal=causal)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                       # (B, T, n_embd)
        x = self.sa_head(x)                                                         # (B, T, head_size)
        logits = self.lm_head(x)                                                    # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Görev 4: Burada bigram'dan farkli olarak, her adimda baglami
        SON self.block_size KARAKTERE KIRPMAK ZORUNDAYIZ (idx_cond).
        Cunku position_embedding_table'in sadece block_size kadar
        pozisyonu var ve Head'in tril maskesi de block_size x block_size
        boyutunda sabitlendi - idx, block_size'dan uzun olursa
        position_embedding_table(torch.arange(T)) tanimsiz bir pozisyon
        ister ve hata verir.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # baglami kirp
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --------------- HAFTA 8 -----------------
class MultiHeadAttention(nn.Module):
    """
    Görev 1: Gecen haftaki tek Head'in (head_size=n_embd) yerine,
    AYNI TOPLAM KAPASITEYI n_head tane kucuk, PARALEL head'e bolup
    birlestiriyoruz.

    Tek head, tum iliski turlerini (bir onceki harf, kelime basi, noktalama
    vs.) AYNI 32 boyutlu uzayda temsil etmek zorunda kaliyordu. Bu bir
    darbogaz. Burada n_head tane bagimsiz Head kuruyoruz (her biri kendi
    query/key/value'sunu ogreniyor, kendi wei'sini hesapliyor), sonra
    ciktilarini KANAL EKSENINDE birlestiriyoruz (torch.cat, dim=-1).
    Boylece n_head * head_size = n_embd'ye geri donuyoruz - toplam
    kapasite ayni, ama artik n_head farkli "bakis acisi" paralel calisiyor.

    nn.ModuleList KULLANMAK ZORUNDAYIZ (duz Python listesi degil) - cunku
    PyTorch, bir modulun alt-modullerini (ve onlarin parametrelerini)
    sadece nn.ModuleList/nn.Sequential gibi konteynerlerle takip edebiliyor.
    Duz bir liste kullansaydik, head'lerin parametreleri model.parameters()
    cagrisinda GORUNMEZDI, optimizer lnlari hic guncellemezdi.

    proj: birlesik ciktiyi (n_head*head_size boyutunda) tekrar n_embd'ye
    projekte eden bir nn.Linear. Su an matematiksel olarak zorunlu degil
    (zaten n_head*head_size=n_embd oluyor) ama onumuzdeki gorevde residual
    baglanti eklerken bu projeksiyonun orada olmasi onemli hale gelecek.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0, causal=True):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout=dropout, causal=causal)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads*head_size) = (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out

class MultiHeadLanguageModel(nn.Module):
    """
    Görev 1: AttentionLanguageModel'in (gecen hafta, tek Head) multi-head
    karsiligi. Tek fark: sa_head yerine sa_heads (MultiHeadAttention).
    Karsilastirmayi adil tutmak icin n_embd, block_size gecen haftayla
    AYNI - tek degisken, head'in tek mi cok mu oldugu.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout=0.0, causal=True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd, n_head'e tam bolunmeli"
        head_size = n_embd // n_head
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout=dropout, causal=causal)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class FeedForward(nn.Module):
    """
    Görev 2: Attention'da token'lar birbiriyle "konusuyordu" (gecmise
    bakip bilgi topluyordu). FeedForward'da her token, topladigi bu
    bilgiyi DIGER TOKENLARDAN BAGIMSIZ olarak kendi basina isliyor. Yani
    basit bir MLP: Linear -> ReLU -> Linear. Ara katmani 4*n_embd'ye
    genisletiyoruz (orijinal "Attention Is All You Need" makalesindeki
    oran) - token, genis bir uzayda "dusunup" tekrar n_embd'ye sikisiyor.
    """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadFeedForwardLanguageModel(nn.Module):
    """
    Görev 2 - Adim 1: multi-head'in ustune FeedForward ekliyoruz, ama
    HENUZ residual baglanti YOK, HENUZ LayerNorm YOK. x, sa_heads'ten
    sonra TAMAMEN DEGISIYOR (toplanmiyor, yerine geciyor), ffwd'den
    sonra da ayni sekilde.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout=0.0, causal=True):
        super().__init__()
        head_size = n_embd // n_head
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout=dropout, causal=causal)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)   # TOPLAMA YOK - x degisiyor
        x = self.ffwd(x)       # TOPLAMA YOK - x degisiyor
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class ResidualLanguageModel(nn.Module):
    """
    Görev 2 - Adim 2: Simdi residual baglanti ekliyoruz (x = x + ...),
    ama HENUZ LayerNorm YOK. Mimari olarak tek fark bir onceki sinifla:
    "=" yerine "+=" mantigi.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout=0.0, causal=True):
        super().__init__()
        head_size = n_embd // n_head
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout=dropout, causal=causal)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = x + self.sa_heads(x)   # RESIDUAL
        x = x + self.ffwd(x)       # RESIDUAL
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Block(nn.Module):
    """
    Görev 2 - Adim 3 (TAM TRANSFORMER BLOGU): residual + LayerNorm +
    multi-head + feedforward, hepsi bir arada. "Pre-norm" konvansiyonunu
    kullaniyoruz (modern GPT'lerin standardi): LayerNorm, sublayer'in
    (attention veya feedforward) GIRDISINE uygulanıyor, ciktisina degil.

        x = x + sa(ln1(x))
        x = x + ffwd(ln2(x))

    Bu sinif, gelecek haftalarda n_layer tane ust uste dizilip gercek
    bir GPT olusturacak yani bu ayni zamanda "LayerNorm eklenmis"
    deney adimi hem de ileride tekrar kullanacagimiz, kalici yapi.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, causal=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout=dropout, causal=causal)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BlockLanguageModel(nn.Module):
    """Görev 2 - Adim 3: tek bir Block'u modele yerlestiriyoruz."""

    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout=0.0, causal=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = Block(n_embd, n_head, block_size, dropout=dropout, causal=causal)
        self.ln_f = nn.LayerNorm(n_embd)  # son katmandan once de bir LayerNorm - standart GPT deseni
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class GPTLanguageModel(nn.Module):
    """
    Görev 3: n_layer tane Block'u ust uste diziyoruz. Artik gercek,
    kucuk bir GPT. nn.Sequential ile Block'lari zincirliyoruz (her
    Block.forward(x) tek bir x alip tek bir x donduruyor, bu yuzden
    Sequential'a direkt verilebiliyor). En sonda bir LayerNorm daha
    (ln_f) standart GPT deseni, lm_head'e girmeden once son bir
    normalize işlemi.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout=0.0, causal=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout=dropout, causal=causal)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx