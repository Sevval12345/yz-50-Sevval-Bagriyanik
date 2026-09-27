import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'görev-1'))
from layers import BatchNorm1d as BatchNorm1dBase

class FlattenConsecutive:
    """
    WaveNet hiyerarşik birleştirme katmanı. 

    Girdideki ardışık 'n' elemanı kanal ekseninde yan yana ekleyerek birleştirir:
        (N, T, C) -> (N, T // n, C * n)

    Model içinde ardışık bloklar halinde kullanıldığında zaman (T) boyutunu
    ikili ağaç (binary tree) yapısıyla kademeli olarak tek bir kök vektöre indirir:
        (N, 8, 24)  -> 1. birleşme -> (N, 4, 48)
        (N, 4, 128) -> 2. birleşme -> (N, 2, 256)
        (N, 2, 128) -> 3. birleşme -> (N, 1, 256)

    Tensörün model boyunca 3 boyutlu akışını ve katmanlar arasındaki boyutsal
    tutarlılığı korumak için T=1 durumunda içeride squeeze uygulanmaz. Tensör
    model çıkışına kadar (N, 1, C) şeklinde taşınır.
    """

    def __init__(self, n=2):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        self.out = x
        return self.out

    def parameters(self):
        return []

class BatchNorm1d(BatchNorm1dBase):
    """
    Hem 2 boyutlu (N, C) hem de 3 boyutlu (N, T, C) tensörleri destekleyen
    Batch Normalization katmanı.

    İstatistik Hesaplama:
    - 2B Girdi (N, C): Ortalama ve varyans yalnızca örnekler (dim=0) üzerinden alınır.
    - 3B Girdi (N, T, C): Ortalama ve varyans hem batch (N) hem de dizi adımları (T)
      boyunca dim=(0, 1) eksenlerinde ortak tek bir dağılım olarak hesaplanır.
      Böylece aynı özellik kanalı, dizinin tüm konumlarında aynı parametrelerle normalize edilir.

    Çıktı ve Durum Güncellemesi:
    - Eğitim modunda (training=True) anlık batch istatistikleri kullanılır,
      hareketli ortalama (running_mean) ve varyans (running_var) güncellenir.
    - Değerlendirme modunda (training=False) çıkarım için kaydedilmiş running istatistikleri kullanılır.
    """

    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()

        return self.out