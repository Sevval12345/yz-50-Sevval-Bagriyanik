import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # varsayılan: leaf node için yapılacak bir şey yok

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # toplama: lokal türev 1, gradyan olduğu gibi dağılır += kullanıyoruz ki aynı değişken birden fazla yerde kullanılırsa gradyanlar üzerine yazılmasın, toplansın
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
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
        # 1. Topolojik sıralamayı kur: her node, kendi childrenı listeye eklendikten sonra eklenir
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

        # 3. Ters topolojik sırayla gez (çıktıdan girdiye doğru) ve her node'da zincir kuralını uygula (_backward çağırarak)
        for node in reversed(topo):
            node._backward()


# Test 1: Nöron örneği (önceki elle hesapladığımızla karşılaştır)
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)

x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + b
o = n.tanh()

o.backward()

print("o.data:", o.data)
print("x1.grad:", x1.grad)   # beklenen: -1.5
print("w1.grad:", w1.grad)   # beklenen: 1.0
print("x2.grad:", x2.grad)   # beklenen: 0.5
print("w2.grad:", w2.grad)   # beklenen: 0.0

print()

# Test 2: Aynı değişkenin birden fazla kullanımı (bug testi)
a = Value(3.0)
b2 = a + a
b2.backward()
print("a.grad (a+a):", a.grad)   # beklenen: 2.0 (1 + 1, üzerine yazılsaydı 1.0 çıkardı)
