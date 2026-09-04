import math


class Value:
    """
    Bir skaler sayıyı sarmalayan, otomatik türev (autograd) destekleyen sınıf.
    Her Value, kendisini üreten Value'ları (_prev) ve hangi işlemden
    çıktığını (_op) saklar - böylece bir computation graph kurulur.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # varsayılan: leaf node için yapılacak bir şey yok

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # toplama: lokal türev 1, gradyan olduğu gibi dağılır
            # += kullanıyoruz ki aynı değişken birden fazla yerde kullanılırsa
            # gradyanlar üzerine yazılmasın, biriksin
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "sadece int/float üs destekleniyor"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # 1. Topolojik sıralamayı kur: her node, kendi çocukları
        #    listeye eklendikten sonra eklenir
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. Taban durumu: çıktının kendisine göre türevi 1
        self.grad = 1.0

        # 3. Ters topolojik sırayla gez (çıktıdan girdiye doğru)
        #    ve her node'da zincir kuralını uygula (_backward çağırarak)
        for node in reversed(topo):
            node._backward()
