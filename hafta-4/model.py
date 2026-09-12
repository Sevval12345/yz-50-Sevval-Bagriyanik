import math
import torch

def init_layers_naive(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Yavas baslatma: tum agirliklar standart sapma 1 ile (torch.randn).
    Bu, hem baslangic loss'unun cok yuksek olmasina hem de tanh'in
    doyuma ulasmasina sebep oluyor (bakınıx -> diagnostics.py).
    """
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, embedding_dim), generator=g)
    W1 = torch.randn((block_size * embedding_dim, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g)
    b2 = torch.randn(vocab_size, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def init_layers_kaiming(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Kaiming init: W1, fan_in'e gore olceklendi (gain=5/3, tanh icin
    onerilen). pre-activation'larin varyansini kontrol altinda tutup
    tanh'in doyuma ulasmasini onluyor. W2 kucuk bir carpanla (0.01) baslatildi,
    (b2=0) baslangic logit'leri kucuk, loss ideal degere [log(vocab_size)]
    yakin cikiyor.
    """
    g = torch.Generator().manual_seed(seed)
    fan_in = block_size * embedding_dim
    gain = 5 / 3

    C = torch.randn((vocab_size, embedding_dim), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g) * (gain / math.sqrt(fan_in))
    b1 = torch.zeros(n_hidden)
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.zeros(vocab_size)

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

def init_layers_batchnorm(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    BatchNorm'lu model icin parametreler. b1 YOK. BatchNorm zaten her
    ornekten ortalamayi cikardigi icin bias'in etkisini iptal ediyor.
    bngain/bnbias: BatchNorm'un ogrenilebilir olcek/kaydirma parametreleri.
    running_mean/running_std: TRAINABLE DEGIL, hareketli ortalama ile
    guncelleniyor (bakınız -> forward_pass_bn).
    """
    g = torch.Generator().manual_seed(seed)
    fan_in = block_size * embedding_dim
    gain = 5 / 3

    C = torch.randn((vocab_size, embedding_dim), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g) * (gain / math.sqrt(fan_in))
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.zeros(vocab_size)
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))

    parameters = [C, W1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    running_mean = torch.zeros((1, n_hidden))
    running_std = torch.ones((1, n_hidden))
    return parameters, running_mean, running_std

def forward_pass(X, C, W1, b1, W2, b2):
    """
    Standart forward pass (yavas ve Kaiming init'li modeller icin):
    1. C[X]: embedding'leri cekiyoruz -> (N, block_size, embedding_dim)
    2. .view(...): DUZLESTIRME -> (N, block_size*embedding_dim)
    3. h = tanh(emb_flat @ W1 + b1): gizli katman
    4. logits = h @ W2 + b2: cikis katmani
    """
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    return logits

def diagnostic_forward(X, parameters):
    """
    forward_pass ile ayni ama h (tanh SONRASI) ile hpreact (tanh'a
    girmeden ONCEKI ham degerler) de donduruyor (doyum teshisi icin).
    """
    C, W1, b1, W2, b2 = parameters
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    hpreact = emb_flat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return logits, h, hpreact

def forward_pass_bn(X, parameters, running_mean, running_std, training=True, momentum=0.001, eps=1e-5):
    """
    BatchNorm'lu forward pass.
    training=True: batch istatistigi kullanilir, running_mean/std
      hareketli ortalamayla guncellenir (no_grad icinde).
    training=False: egitim boyunca biriktirilen running_mean/std
      kullanilir (tahmin/dev/test icin).
    """
    C, W1, W2, b2, bngain, bnbias = parameters
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    hpreact = emb_flat @ W1  # bias yok, BatchNorm zaten merkezliyor

    if training:
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        with torch.no_grad():
            running_mean *= (1 - momentum)
            running_mean += momentum * bnmeani
            running_std *= (1 - momentum)
            running_std += momentum * bnstdi
    else:
        bnmeani = running_mean
        bnstdi = running_std

    hpreact_norm = (hpreact - bnmeani) / (bnstdi + eps)
    hpreact_final = bngain * hpreact_norm + bnbias
    h = torch.tanh(hpreact_final)
    logits = h @ W2 + b2
    return logits