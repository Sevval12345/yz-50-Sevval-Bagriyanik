import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))

import torch
import torch.nn.functional as F

from dataset import read_words, build_vocab, build_dataset, split_words   # görev-1'den, aynen
from train import train, plot_loss_curve                                   # görev-1'den, aynen
from model2 import build_wavenet_model                                      # görev-3'ün kendi model.py'si

if __name__ == "__main__":
    words = read_words(os.path.join(os.path.dirname(__file__), '..', 'görev-1', 'names.txt'))
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 8, 24, 128

    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=block_size)

    model = build_wavenet_model(vocab_size, embedding_dim, block_size, n_hidden)

    print("\nWaveNet katmanları ve çıktı şekilleri (4 örneklik bir batch ile):")
    Xb, Yb = Xtr[:4], Ytr[:4]
    x = Xb
    for layer in model.layers:
        x = layer(x)
        print(f"  {layer.__class__.__name__:<20} çıktı şekli: {tuple(x.shape)}")
    print()

    """"
    ŞEKİL NEDEN BÖYLE?: FlattenConsecutive her calistiginda T eksenini ikiye bolup C eksenine
    katliyor. Bu yuzden T her katmanda yariya iniyor: 8, sonra 4, sonra 2,
    son katmanda 1'e dusup squeeze ediliyor. C ise tam tersi yonde
    buyuyor - birlesen iki grubun kanallari yan yana dizildigi icin ikiye
    katlaniyor (24->48, 128->256), ardindan gelen Linear katmani bu genis
    vektoru tekrar n_hidden boyutuna indiriyor. Yani agac yapisinin her
    seviyesinde "daha az grup, daha zengin bilgi" mantigi isliyor.
    """

    model, lossi = train(Xtr, Ytr, Xdev, Ydev, model,
                          learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000)

    with torch.no_grad():
        final_logits = model(Xdev)
        if final_logits.ndim == 3 and final_logits.shape[1] == 1:
            final_logits = final_logits.squeeze(1)
        final_dev_loss = F.cross_entropy(final_logits, Ydev).item()

    print(f"\nNihai dev loss: {final_dev_loss:.4f}")