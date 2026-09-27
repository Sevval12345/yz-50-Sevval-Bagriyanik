import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "görev-1"))
from layers import Embedding, Linear, Sequential, Tanh
from wavenet_layers import BatchNorm1d, FlattenConsecutive

def build_wavenet_model(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Karakter düzeyinde dil modelleme için hiyerarşik WaveNet mimarisi oluşturur.

    Mimari Yapısı:
    - Girdi: (N, block_size) boyutunda karakter indeksleri.
    - Embedding: Karakterleri (N, block_size, embedding_dim) vektörlerine dönüştürür.
    - Hiyerarşik Bloklar (3 kademeli ikili ağaç):
        1. Blok: Ardışık 2 harfi birleştirir (n=2), kanal boyutu 2 katına çıkar.
                 Linear projeksiyonu ile n_hidden boyutuna harmanlanır,
                 BatchNorm1d ile normalize edilip Tanh aktivasyonundan geçer.
        2. Blok: 4'er harflik öbekleri ikişerli birleştirir, tekrar n_hidden boyutuna indirir.
        3. Blok: Kalan 2 öbeği birleştirerek tüm bağlamı tek bir kök vektörde (n_hidden) toplar.
    - Çıkış Katmanı: Nihai bağlam temsilini alfabe boyutu (vocab_size) kadar logit değerine eşler.

    Parametreler:
        - vocab_size (int): Alfabedeki toplam karakter/token sayısı.
        - embedding_dim (int): Karakter başına gömme vektörü boyutu.
        - block_size (int): Bağlam uzunluğu (ikili ağaç için 2'nin kuvveti olmalıdır, örn. 8).
        - n_hidden (int): Ara katmanlardaki gizli nöron/özellik sayısı.
        - seed (int): Ağırlık ilklendirmesi için rastgelelik tohumu.

    Döndürdüğü:
        Sequential: Katmanları sıralı olarak çalıştıran model nesnesi.
    """
    torch.manual_seed(seed)

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        FlattenConsecutive(2), Linear(embedding_dim * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])

    # Başlangıçta kayıp değerini ideale yakın başlatmak için çıkış katmanı ağırlıkları ölçeklenir
    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    for p in model.parameters():
        p.requires_grad = True

    total_params = sum(p.nelement() for p in model.parameters())
    print(f"Toplam parametre sayisi: {total_params}")

    return model