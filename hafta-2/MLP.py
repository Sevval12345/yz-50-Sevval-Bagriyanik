from layer import Layer


class MLP:
    """
    Multi-Layer Perceptron: katmanları art arda dizer.
    Bir katmanın çıktısı bir sonrakinin girdisi olur.
    nouts: her katmanın kaç nörona sahip olacağını belirten liste.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
