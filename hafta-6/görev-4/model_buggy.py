import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-3'))

from layers import Linear, Tanh, Embedding, Sequential
# 3B girdilerde normalizasyonu yalnızca dim=0 (batch) ekseninde uygulayan temel BatchNorm1d:
from layers import BatchNorm1d as SingleAxisBatchNorm1d
from wavenet_layers import FlattenConsecutive

def build_wavenet_model_buggy(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Uzaklaştırma ve karşılaştırma amaçlı hiyerarşik WaveNet mimarisi oluşturur.

    Bu varyasyonda ara katmanlardaki Batch Normalization işlemi, 3 boyutlu
    (N, T, C) tensörler için dizi/zaman (T) eksenini dahil etmeden yalnızca
    örnekler (dim=0) üzerinden çalıştırılır. Bu durum, her zaman adımının
    bağımsız bir dağılımmış gibi ele alınmasına yol açar ve normalizasyon
    istatistiklerinin dizi genelinde paylaşılmadığı durumu temsil eder.

    Parametreler:
        - vocab_size (int): Alfabedeki toplam karakter sayısı.
        - embedding_dim (int): Karakter başına gömme vektörü boyutu.
        - block_size (int): Bağlam uzunluğu.
        - n_hidden (int): Ara katmanlardaki gizli nöron sayısı.
        - seed (int): Ağırlık ilklendirmesi için rastgelelik tohumu.

    Döndürür:
        Sequential: Karşılaştırma modelini temsil eden katman dizisi.
    """
    torch.manual_seed(seed)

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        FlattenConsecutive(2), Linear(embedding_dim * 2, n_hidden, bias=False), SingleAxisBatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), SingleAxisBatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), SingleAxisBatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])

    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    for p in model.parameters():
        p.requires_grad = True

    return model