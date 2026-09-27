import torch
from layers import Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential

def build_model(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Modeli tek bir Sequential olarak kuruyoruz. Egitim
    dongusu artik "Linear'e W1 de, sonra BatchNorm'a ver, sonra
    tanh'a ver" diye elle yazmiyor sadece model(Xb) diyor.

    Katman sirasi:
    - Embedding: karakter indeksini vektore cevirir
    - Flatten: block_size*embedding_dim'e duzlestirir
    - Linear (bias=False): BatchNorm zaten kendi bias'ini
    icerdigi icin, bu Linear'in bias'i gereksizdi.Bunu kapatiyoruz
    (gecen hafta b1'in "sifira gittigini" görmustuk, bu
    sefer onu bastan hic eklemiyoruz).
    - BatchNorm1d + Tanh: gizli katman
    - Linear: cikis katmani, logits uretir
    """
    torch.manual_seed(seed)

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        Flatten(),
        Linear(embedding_dim * block_size, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size),
    ])

    # Cikis katmaninin agirliklarini kucultuyoruz. Gecen haftalardaki "başlangic loss'unu ideale yaklastirma" mantigi
    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    for p in model.parameters():
        p.requires_grad = True

    total_params = sum(p.nelement() for p in model.parameters())
    print(f"Toplam parametre sayisi: {total_params}")

    return model