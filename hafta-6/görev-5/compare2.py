import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))

import torch
import torch.nn.functional as F
from dataset import read_words, build_vocab, build_dataset, split_words
from model import build_model
from train import train

def run_flat_model(Xtr, Ytr, Xdev, Ydev, vocab_size, block_size, embedding_dim, n_hidden,
                    learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000):
    """
    Tek bir konfigurasyonu (belirli block_size ile) egitip, parametre
    sayisini ve nihai dev loss'unu donduren yardimci fonksiyon.
    Mimari hep ayni (Embedding->Flatten->Linear->BatchNorm1d->Tanh->Linear).
    Sadece block_size degisiyor.
    """
    model = build_model(vocab_size, embedding_dim, block_size, n_hidden)
    n_params = sum(p.nelement() for p in model.parameters())

    model, lossi = train(Xtr, Ytr, Xdev, Ydev, model,
                          learning_rate=learning_rate, batch_size=batch_size,
                          steps=steps, eval_every=eval_every)

    with torch.no_grad():
        final_logits = model(Xdev)
        final_dev_loss = F.cross_entropy(final_logits, Ydev).item()

    return n_params, final_dev_loss, model

if __name__ == "__main__":
    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    embedding_dim, n_hidden = 10, 200

    train_words, dev_words, test_words = split_words(words)

    results = {}

    print("=" * 60)
    print("BASELINE: block_size = 3 (hafta 4'teki model)")
    print("=" * 60)
    Xtr3, Ytr3 = build_dataset(train_words, stoi, block_size=3)
    Xdev3, Ydev3 = build_dataset(dev_words, stoi, block_size=3)
    n_params_3, dev_loss_3, _ = run_flat_model(
        Xtr3, Ytr3, Xdev3, Ydev3, vocab_size, block_size=3,
        embedding_dim=embedding_dim, n_hidden=n_hidden)
    results['block_size=3'] = (n_params_3, dev_loss_3)
    print(f"\nSonuc: parametre={n_params_3}, dev_loss={dev_loss_3:.4f}\n")

    print("=" * 60)
    print("GOREV 2: block_size = 8 (ayni duz mimari)")
    print("=" * 60)
    Xtr8, Ytr8 = build_dataset(train_words, stoi, block_size=8)
    Xdev8, Ydev8 = build_dataset(dev_words, stoi, block_size=8)
    n_params_8, dev_loss_8, _ = run_flat_model(
        Xtr8, Ytr8, Xdev8, Ydev8, vocab_size, block_size=8,
        embedding_dim=embedding_dim, n_hidden=n_hidden)
    results['block_size=8'] = (n_params_8, dev_loss_8)
    print(f"\nSonuc: parametre={n_params_8}, dev_loss={dev_loss_8:.4f}\n")

    print("=" * 60)
    print("KARSILASTIRMA (bu tablo, sonraki gorevler icin referans)")
    print("=" * 60)
    print(f"{'Konfigurasyon':<18}{'Parametre':<15}{'Dev Loss':<12}")
    print("-" * 45)
    for name, (n_params, dev_loss) in results.items():
        print(f"{name:<18}{n_params:<15}{dev_loss:<12.4f}")

    param_artis = n_params_8 - n_params_3
    param_artis_yuzde = (param_artis / n_params_3) * 100
    loss_degisim = dev_loss_8 - dev_loss_3

    print()
    print(f"Parametre artisi: {param_artis} (+%{param_artis_yuzde:.1f})")
    if loss_degisim < 0:
        print(f"Dev loss dususu: {abs(loss_degisim):.4f} (iyilesme)")
    else:
        print(f"Dev loss artisi: {loss_degisim:.4f} (kotulesme)")