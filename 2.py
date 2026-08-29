# 1. Basit ifade

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

# İfadeyi kuralım
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f

print("L.data:", L.data)

# Manuel Backpropagation 

# 1. Taban durumu: dL/dL = 1
L.grad = 1.0

# 2. L = d * f  ->  dL/dd = f, dL/df = d
d.grad = f.data * L.grad
f.grad = d.data * L.grad

# 3. d = e + c  ->  toplama lokal türevi 1, gradyan olduğu gibi dağılır
c.grad = 1.0 * d.grad
e.grad = 1.0 * d.grad

# 4. e = a * b  ->  dL/da = b, dL/db = a
a.grad = b.data * e.grad
b.grad = a.data * e.grad

# Sonuçlar
print("a.grad:", a.grad)
print("b.grad:", b.grad)
print("c.grad:", c.grad)
print("e.grad:", e.grad)
print("d.grad:", d.grad)
print("f.grad:", f.grad)

# 2. Tek bir neuron
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        return out

# Nöron ifadesini kuralum
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)   # elle takip edebilmek için "güzel" seçilmiş sayı (ai'dan yardım aldım)

x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + b
o = n.tanh()

print("o.data:", o.data)

# Manuel Backpropagation 
# 1. Taban durumu: do/do = 1
o.grad = 1.0

# 2. o = tanh(n)  ->  do/dn = 1 - tanh(n)^2 = 1 - o.data^2
n.grad = (1 - o.data ** 2) * o.grad

# 3. n = x1w1 + x2w2 + b  ->  toplama lokal türevi 1, gradyan%% olduğu gibi dağılır
x1w1.grad = 1.0 * n.grad
x2w2.grad = 1.0 * n.grad
b.grad = 1.0 * n.grad

# 4. x1w1 = x1 * w1  ->  çarpma: karşı tarafın data'sı ile çarp
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

# 5. x2w2 = x2 * w2
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

# Sonuçlar
print("n.grad:", n.grad)
print("x1w1.grad:", x1w1.grad)
print("x2w2.grad:", x2w2.grad)
print("b.grad:", b.grad)
print("x1.grad:", x1.grad)
print("w1.grad:", w1.grad)
print("x2.grad:", x2.grad)
print("w2.grad:", w2.grad)