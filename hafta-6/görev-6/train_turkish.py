import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-3'))

import torch
import torch.nn.functional as F

from dataset import read_words, build_vocab, build_dataset, split_words
from train import train
from model import build_model
from model2 import build_wavenet_model


def sample_names(model, itos, block_size, n=20, seed=42):
    """
    Egitilmis modelden isim uretmek icin. makemore'daki
    ornekleme mantigi: '.' ile dolu bir baglamla basla, model her
    adimda bir sonraki karakterin olasilik dagilimini (softmax)
    verir, o dagilimdan ORNEKLEME yaparak (argmax degil, cesitlilik
    icin) bir sonraki karakteri sec, baglsami bir kaydir, '.' gelene
    kadar (veya asiri uzarsa) devam et.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            x = torch.tensor([context])
            logits = model(x)
            # WaveNet çıktısı (1, 1, vocab_size) ise (1, vocab_size) boyutuna indirgenir:
            if logits.ndim == 3 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
            if len(out) > 30:  # Güvenlik sınırı (sonsuz döngüyü engellemek için)
                break
        names.append(''.join(out))
    model.train()
    return names

if __name__ == "__main__":
    words = read_words(os.path.join(os.path.dirname(__file__), 'turkce_isimler.txt'))
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    print(f"Kelime sayisi: {len(words)}, alfabe boyutu (vocab_size): {vocab_size}")

    train_words, dev_words, test_words = split_words(words)

    # ---- HAFTA 4 DÜZ MLP: block_size=3 ----
    print("\n" + "=" * 60)
    print("HAFTA 4 TURKCE MLP (baseline): block_size=3")
    print("=" * 60)
    block_size_mlp = 3
    Xtr_mlp, Ytr_mlp = build_dataset(train_words, stoi, block_size=block_size_mlp)
    Xdev_mlp, Ydev_mlp = build_dataset(dev_words, stoi, block_size=block_size_mlp)

    model_mlp = build_model(vocab_size, embedding_dim=10, block_size=block_size_mlp, n_hidden=200)
    model_mlp, _ = train(Xtr_mlp, Ytr_mlp, Xdev_mlp, Ydev_mlp, model_mlp,
                          learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000)
    
    model_mlp.eval()
    with torch.no_grad():
        logits_mlp = model_mlp(Xdev_mlp)
        if logits_mlp.ndim == 3 and logits_mlp.shape[1] == 1:
            logits_mlp = logits_mlp.squeeze(1)
        dev_loss_mlp = F.cross_entropy(logits_mlp, Ydev_mlp).item()
    n_params_mlp = sum(p.nelement() for p in model_mlp.parameters())

    # ---- WAVENET: block_size=8 ----
    print("\n" + "=" * 60)
    print("WAVENET: block_size=8")
    print("=" * 60)
    block_size_wn = 8
    Xtr_wn, Ytr_wn = build_dataset(train_words, stoi, block_size=block_size_wn)
    Xdev_wn, Ydev_wn = build_dataset(dev_words, stoi, block_size=block_size_wn)

    model_wn = build_wavenet_model(vocab_size, embedding_dim=24, block_size=block_size_wn, n_hidden=128)
    model_wn, _ = train(Xtr_wn, Ytr_wn, Xdev_wn, Ydev_wn, model_wn,
                         learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000)
    
    model_wn.eval()
    with torch.no_grad():
        logits_wn = model_wn(Xdev_wn)
        if logits_wn.ndim == 3 and logits_wn.shape[1] == 1:
            logits_wn = logits_wn.squeeze(1)
        dev_loss_wn = F.cross_entropy(logits_wn, Ydev_wn).item()
    n_params_wn = sum(p.nelement() for p in model_wn.parameters())

    # ---- Karşılaştırma ----
    print("\n" + "=" * 60)
    print("KARSILASTIRMA (Turkce isimler)")
    print("=" * 60)
    print(f"{'Model':<28}{'Parametre':<12}{'Dev Loss':<10}")
    print("-" * 50)
    print(f"{'Hafta 4 MLP (block=3)':<28}{n_params_mlp:<12}{dev_loss_mlp:<10.4f}")
    print(f"{'WaveNet (block=8)':<28}{n_params_wn:<12}{dev_loss_wn:<10.4f}")
    print(f"\nDev loss farki: {dev_loss_mlp - dev_loss_wn:+.4f} (pozitifse WaveNet daha iyi)")

    # ---- Üretilen İsimler ----
    print("\n" + "=" * 60)
    print("HAFTA 4 MLP'DEN URETILEN ISIMLER")
    print("=" * 60)
    for name in sample_names(model_mlp, itos, block_size_mlp, n=20, seed=42):
        print(" ", name)

    print("\n" + "=" * 60)
    print("WAVENET'TEN URETILEN ISIMLER")
    print("=" * 60)
    for name in sample_names(model_wn, itos, block_size_wn, n=20, seed=42):
        print(" ", name)

"""
Turkce veri setinde WaveNet, duz MLP'den daha kotu sonuc verdi
(2.2311 -> 2.4160). Egitim eğrisine bakinca sebebi belli: train loss
dusmeye devam ederken (1.89 -> 1.44) dev loss yukseliyor (2.195 ->
2.416) Yani ezberlerliyor!! Turkce liste, ingilizce
isim listesinin yaklasik onda biri buyuklugunde, ama WaveNet
77 bin parametreyle (duz modelin x6) bu kucuk veriye
gore fazla buyuk kaliyor. Yani baglam ve model kapasitesini artirmak
ancak yeterli veri varsa ise yariyor kucuk veride tam tersi etki
yapabiliyor.
"""