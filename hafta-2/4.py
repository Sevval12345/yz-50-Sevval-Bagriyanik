import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
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
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# ----------------------------------------------------------
# 1: exp, +, -, / kullanarak tanh'ı parçala ve tek-node tanh() ile aynı gradyanları aldığını göster

def make_neuron_fused_tanh():
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    n = x1w1 + x2w2 + b
    o = n.tanh()  # tek node olarak tanh
    return x1, x2, w1, w2, b, o

def make_neuron_broken_tanh():
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    n = x1w1 + x2w2 + b
    # tanh'ı atomik parçalara böl: (e^2x - 1) / (e^2x + 1)
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    return x1, x2, w1, w2, b, o

x1, x2, w1, w2, b, o = make_neuron_fused_tanh()
o.backward()
print("[Tek-node tanh]")
print(f"o.data = {o.data}")
print(f"x1.grad={x1.grad}, w1.grad={w1.grad}, x2.grad={x2.grad}, w2.grad={w2.grad}")

x1b, x2b, w1b, w2b, bb, ob = make_neuron_broken_tanh()
ob.backward()
print("\n[Parçalanmış tanh: exp, +, -, /]")
print(f"o.data = {ob.data}")
print(f"x1.grad={x1b.grad}, w1.grad={w1b.grad}, x2.grad={x2b.grad}, w2.grad={w2b.grad}")

# -----------------------------------------------------------
# 2: Üç yöntemi karşılaştır
#   1) backward() (analitik / otomatik gradyan)
#   2) numerical derivative (geçen haftaki h ile yaklaşık türev)
#   3) PyTorch autograd

def loss_from_w1(w1_value):
    x1 = Value(2.0); x2 = Value(0.0)
    w1 = Value(w1_value); w2 = Value(1.0)
    b = Value(6.8813735870195432)
    n = x1 * w1 + x2 * w2 + b
    return n.tanh().data


# 1) backward()
x1 = Value(2.0); x2 = Value(0.0)
w1 = Value(-3.0); w2 = Value(1.0)
b = Value(6.8813735870195432)
n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()
grad_backward = w1.grad

# 2) Numerical derivative
h = 0.0001
loss1 = loss_from_w1(-3.0)
loss2 = loss_from_w1(-3.0 + h)
grad_numerical = (loss2 - loss1) / h

# 3) PyTorch autograd
import torch

x1_t = torch.Tensor([2.0]).double(); x1_t.requires_grad = True
x2_t = torch.Tensor([0.0]).double(); x2_t.requires_grad = True
w1_t = torch.Tensor([-3.0]).double(); w1_t.requires_grad = True
w2_t = torch.Tensor([1.0]).double(); w2_t.requires_grad = True
b_t = torch.Tensor([6.8813735870195432]).double(); b_t.requires_grad = True

n_t = x1_t * w1_t + x2_t * w2_t + b_t
o_t = torch.tanh(n_t)
o_t.backward()
grad_pytorch = w1_t.grad.item()

print(f"\nw1'e göre gradyan (dL/dw1), üç yöntemle:")
print(f"  1) backward()          : {grad_backward:.6f}")
print(f"  2) numerical derivative : {grad_numerical:.6f}")
print(f"  3) PyTorch autograd     : {grad_pytorch:.6f}")
