import random
from value import Value


class Neuron:
    """
    Tek bir nöron: nin tane girdi alır, kendi ağırlıkları ve bias'ıyla
    ağırlıklı toplamı hesaplar, tanh ile sıkıştırır.
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w*x'lerin toplamına bias'ı ekleyerek başlıyoruz (sum'ın start parametresi)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
