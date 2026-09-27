import torch

class Linear:
    """
    torch.nn.Linear'in kendi versiyonumuz. y = x @ W (+ b).

    fan_in**-0.5 ile olcekliyoruz. Gecen hafta kullandigimiz
    Kaiming init'in ayni mantigi, burada dogrudan katmanin kendi
    __init__'ine gomuluyor.
    """

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    """
    torch.nn.BatchNorm1d'in kendi versiyonumuz.

    training=True: batch istatistigini kullanir, running_mean/var'i
    hareketli ortalamayla gunceller (gecen haftanşn mantigiyla -> ayni).
    training=False: running_mean/var kullanir (tahmin/dev/test).

    -gamma (bngain) ve beta (bnbias): ogrenilebilir, parameters()
    listesine giriyor.
    -running_mean/running_var: OGRENILEBILIR DEGIL!!!!!! parameters()'a
    girmiyor, gradient descent'le degil hareketli ortalamayla
    guncelleniyor.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    """torch.nn.Tanh'in kendi versiyonumuz. Parametresi yok."""

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:
    """
    torch.nn.Embedding'in kendi versiyonumuz. Karakter indekslerini
    ogrenilebilir vektorlere ceviren tablo -> gecen haftalarda yazdığımız C yani.
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]

class Flatten:
    """
    torch.nn.Flatten'in kendi (basitlestirilmis) versiyonumuz.
    (N, block_size, embedding_dim) -> (N, block_size*embedding_dim)
    Parametresi yok.
    """

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

class Sequential:
    """
    torch.nn.Sequential'in kendi versiyonumuz. Bir katman listesi
    tutuyor, forward'i basit: girdiyi ilk katmana ver, ciktisini
    ikinciye ver. Bu sayede egitim dongusu artik katmanlarin
    ISIMLERINI BILMEDEN, sadece model(x) diyerek calisabiliyor.
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # Her katmanin kendi parametrelerini topluyoruz. Tek bir duz liste halinde donduruyoruz.
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        """Egitim moduna geciriyor: BatchNorm1d katmanlari batch istatistigi kullanacak."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """Tahmin moduna geciriyor: BatchNorm1d katmanlari running istatistik kullanacak."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False