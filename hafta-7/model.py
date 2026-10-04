import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """
    Görev 3: Tek bir self-attention head.

    key, query, value: ucu de x'ten nn.Linear (bias'siz) ile uretilen
    kucuk (head_size boyutunda) projeksiyonlar.
    - query: "ben ne ariyorum?"
    - key:   "ben ne sunuyorum?"
    - value: "beni secersen sana verecegim bilgi" (x'in kendisi degil,
    ayri bir ogrenilmis projeksiyon)

    wei = q @ k.T: her token ciftinin ilgi puani (B,T,T). head_size'in
    karekoku ile olcekliyoruz (asagidaki demo'da neden gerekli oldugunu
    sayiyla gosteriyoruz). tril ile GELECEK pozisyonlari -inf yapip
    softmax'a veriyoruz boylece bir token sadece KENDISINDEN ONCEKI
    (ve kendisi dahil) tokenlara bakabiliyor, gelecegi goremiyor.

    forward() ciktisinin yaninda self.wei'yi de SAKLIYORUZ (hafta 4 ve 6'daki
    kendi katmanlarimizin self.out'u sakladigi gibi). Boylece disaridan
    "bu token hangilerine ne kadar bakti?" diye inceleyebiliyoruz.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0, causal=True):
        super().__init__()
        self.head_size = head_size
        self.causal = causal  # Not: Ek-b: False yapilirsa tril maskesi uygulanmaz
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
        # self.causal=False ise: Ek-b: maske yo, her token gelecekteki tokenlara da bakabiliyor (kendi hedefi de dahil).
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        self.wei = wei      # (B, T, T) - her tokenin hangi tokenlara ne kadar baktigi

        v = self.value(x)    # (B, T, head_size)
        out = wei @ v         # (B, T, head_size)
        return out

class BigramLanguageModel(nn.Module):
    """
    Görev 1: Hafta 3'teki bigram modelinin (27x27 agirlik matrisi, kendi
    elimizle yazdigimiz) torch.nn diliyle yazilmis hali. Artik kendi
    Linear/Embedding siniflarimizi degil, gercek torch.nn.Module'u
    kullaniyoruz.

    Tek katman: nn.Embedding(vocab_size, vocab_size). Bu, her karakter
    icin DOGRUDAN bir sonraki karakterin logit'lerini tutan bir tablo
    yani token_embedding_table[i] = i'den sonra hangi karakterin
    gelecegine dair (henuz egitilmemis) tahmin. Baglam onemli degil,
    sadece son karaktere bakiyor (bigram = 2'li, bir onceki + simdiki).

    Bu, ileride attention ekleyecegimiz modelin "karsilastirma tabani".
    Attention'in gercekten fayda sagladigini gormek icin once
    attention'siz halin val loss'unu bilmemiz gerekiyor.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T): B ornek, T uzunlugunda baglam (tam sayi indeksleri)
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
        olasilik dagilimini al, ornekleme yap (argmax degil, cesitlilik
        icin), uretilen karakteri baglama ekle, tekrarla. Bigram modelde
        baglamin uzunlugu onemli degil (sadece son karaktere bakiyor),
        ama fonksiyonu genel yaziyoeuz ki ileride attention'li modelde de
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
    Göreb 4: Head'i (görev 3) bigram modelinin (görev 1) icine koyuyoruz.

    Eski BigramLanguageModel: tek katman, dogrudan vocab_size'a cikis
    yapan nn.Embedding. Aradan hicbir sey gecmiyordu.

    Bu model:
      1. token_embedding_table: karakteri n_embd boyutlu bir vektore cevir
      2. position_embedding_table: pozisyon bilgisini ekle (görev 3)
      3. sa_head: self-attention ile gecmise bakarak vektorleri harmanla
      4. lm_head: harmanlanmis vektoru tekrar vocab_size boyutuna (logit)
         cevirmek icin son bir nn.Linear katmani

    Block_size, hem position_embedding_table'in boyutunu hem de Head
    icindeki tril maskesinin boyutunu belirliyor. Bu yuzden model
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
        son self.block_size karaktere kırpmak zorundayız (idx_cond).
        Cunku position_embedding_table'in sadece block_size kadar
        pozisyonu var ve Head'in tril maskesi de block_size x block_size
        boyutunda sabitlendi. idx, block_size'dan uzun olursa
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