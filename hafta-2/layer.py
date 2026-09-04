from neuron import Neuron


class Layer:
    """
    Bir katman: nout tane bağımsız Neuron'dan oluşur.
    Her nöron aynı girdiyi (x) alır ama kendi ağırlıklarını kullanır.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
