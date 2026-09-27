import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-3'))

import torch
import torch.nn.functional as F

from dataset import read_words, build_vocab, build_dataset, split_words   
from train import train                                                    
from compare import run_flat_model                                         
from model2 import build_wavenet_model                                     


def run_wavenet_model(Xtr, Ytr, Xdev, Ydev, vocab_size, block_size, embedding_dim, n_hidden,
                       learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000):
    """
    run_flat_model'in (görev-2) WaveNet karsiligi. Ayni
    mantik, tek fark build_model yerine build_wavenet_model
    kullanilmasi. Modeli kur, egit, parametre sayisini ve nihai
    dev loss'u dondur.
    """
    model = build_wavenet_model(vocab_size, embedding_dim, block_size, n_hidden)
    n_params = sum(p.nelement() for p in model.parameters())

    model, lossi = train(Xtr, Ytr, Xdev, Ydev, model,
                          learning_rate=learning_rate, batch_size=batch_size,
                          steps=steps, eval_every=eval_every)

    with torch.no_grad():
        model.eval()
    with torch.no_grad():
        logits = model(Xdev)
        if logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        final_dev_loss = F.cross_entropy(logits, Ydev).item()

    return n_params, final_dev_loss, model

if __name__ == "__main__":
    names_path = os.path.join(os.path.dirname(__file__), '..', 'görev-1', 'names.txt')
    words = read_words(names_path)
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    train_words, dev_words, test_words = split_words(words)

    results = []

    # --- Baglam 3, duz MLP: hafta 4'teki kucuk model, degismedi ---
    print("=" * 60)
    print("Bağlam 3, düz MLP")
    print("=" * 60)
    Xtr3, Ytr3 = build_dataset(train_words, stoi, block_size=3)
    Xdev3, Ydev3 = build_dataset(dev_words, stoi, block_size=3)
    n3, loss3, _ = run_flat_model(Xtr3, Ytr3, Xdev3, Ydev3, vocab_size,
                                   block_size=3, embedding_dim=10, n_hidden=200)
    results.append(("Bağlam 3, düz MLP", n3, loss3))

    # --- Baglam 8, duz MLP: ayni mimari, sadece block_size buyudu (gorev 2) ---
    print("\n" + "=" * 60)
    print("Bağlam 8, düz MLP")
    print("=" * 60)
    Xtr8, Ytr8 = build_dataset(train_words, stoi, block_size=8)
    Xdev8, Ydev8 = build_dataset(dev_words, stoi, block_size=8)
    n8, loss8, _ = run_flat_model(Xtr8, Ytr8, Xdev8, Ydev8, vocab_size,
                                   block_size=8, embedding_dim=10, n_hidden=200)
    results.append(("Bağlam 8, düz MLP", n8, loss8))

    # --- Baglam 8, WaveNet: GOREV 5 - modeli buyuttuk. embedding_dim 10 -> 24, n_hidden 200 -> 128 (3 katmana yayilmis, her FlattenConsecutive kanal sayisini 2 katina cikardigi icin toplam kapasite dar bir n_hidden ile bile duz modelden fazla). ---
    print("\n" + "=" * 60)
    print("Bağlam 8, WaveNet (büyütülmüş: embedding_dim=24, n_hidden=128)")
    print("=" * 60)
    n_wn, loss_wn, _ = run_wavenet_model(Xtr8, Ytr8, Xdev8, Ydev8, vocab_size,
                                          block_size=8, embedding_dim=24, n_hidden=128)
    results.append(("Bağlam 8, WaveNet", n_wn, loss_wn))

    print("\n" + "=" * 60)
    print("SONUC TABLOSU (GOREV 5)")
    print("=" * 60)
    print(f"{'Konfigurasyon':<22}{'Parametre':<14}{'Dev Loss':<10}")
    print("-" * 46)
    for name, n_params, dev_loss in results:
        print(f"{name:<22}{n_params:<14}{dev_loss:<10.4f}")