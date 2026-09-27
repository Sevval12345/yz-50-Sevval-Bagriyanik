import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-3'))

import torch
import torch.nn.functional as F

from dataset import read_words, build_vocab, build_dataset, split_words   
from train import train                                                    
from model_buggy import build_wavenet_model_buggy                          
from model2 import build_wavenet_model                                     

"""
Sorun su: x.mean(0) ifadesi sadece batch (N) eksenini yok ediyor.
2 boyutlu girdide (N, C) bu yeterliydi cunku geriye zaten tek eksen
(C) kaliyordu. Ama 3 boyutlu girdide (N, T, C) T eksenini es geciyoruz.
Yani her T konumu (her harf grubu) kendi ayri istatistigiyle
normalize ediliyır, oysa hepsinin AYNI kanal istatistigini paylasmasi
gerekiyor. Duzeltme: ortalama ve varyansi dim=(0,1) uzerinden almak,
yani hem batch hem T eksenini birlikte eritmek. running_mean'in sekli
bozulmasin diye de squeeze() ile (C,) haline geri getirdim.
"""

if __name__ == "__main__":
    names_path = os.path.join(os.path.dirname(__file__), '..', 'görev-1', 'names.txt')
    words = read_words(names_path)
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 8, 24, 128
    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=block_size)

    print("=" * 60)
    print("DUZELTME ONCESI: BatchNorm1d sadece dim=0 (3B'de yanlis)")
    print("=" * 60)
    model_buggy = build_wavenet_model_buggy(vocab_size, embedding_dim, block_size, n_hidden)
    model_buggy, _ = train(Xtr, Ytr, Xdev, Ydev, model_buggy,
                            learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000)
    model_buggy.eval()
    with torch.no_grad():
        logits_buggy = model_buggy(Xdev)
        if logits_buggy.ndim == 3 and logits_buggy.shape[1] == 1:
            logits_buggy = logits_buggy.squeeze(1)
        dev_loss_buggy = F.cross_entropy(logits_buggy, Ydev).item()

    print("\n" + "=" * 60)
    print("DUZELTME SONRASI: BatchNorm1d dim=(0,1) (3B'de dogru)")
    print("=" * 60)
    model_fixed = build_wavenet_model(vocab_size, embedding_dim, block_size, n_hidden)
    model_fixed, _ = train(Xtr, Ytr, Xdev, Ydev, model_fixed,
                            learning_rate=0.1, batch_size=32, steps=20000, eval_every=5000)
    model_fixed.eval()
    with torch.no_grad():
        logits_fixed = model_fixed(Xdev)
        if logits_fixed.ndim == 3 and logits_fixed.shape[1] == 1:
            logits_fixed = logits_fixed.squeeze(1)
        dev_loss_fixed = F.cross_entropy(logits_fixed, Ydev).item()

    print("\n" + "=" * 60)
    print("KARSILASTIRMA")
    print("=" * 60)
    print(f"{'Versiyon':<22}{'Dev Loss':<12}")
    print("-" * 34)
    print(f"{'Duzeltme oncesi':<22}{dev_loss_buggy:<12.4f}")
    print(f"{'Duzeltme sonrasi':<22}{dev_loss_fixed:<12.4f}")
    print(f"\nFark: {dev_loss_buggy - dev_loss_fixed:+.4f}")