import math
from value import Value
from neuron import Neuron
from layer import Layer
from mlp import MLP


def gorev_1():
    """
    GOREV 1: Value sinifi - toplama ve carpma (19:09 - 32:10)
    Her yeni Value, kendisini ureten Value'lari (_prev) ve hangi
    islemden ciktigini (_op) saklar - boylece bir computation graph kurulur.
    """
    print("=" * 60)
    print("GOREV 1: Value sinifi - toplama ve carpma")
    print("=" * 60)

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    e = a * b
    d = e + c

    print("d.data:", d.data)
    print("d._prev:", d._prev)
    print("d._op:", d._op)
    print()


def gorev_2():
    """
    GOREV 2: Manuel backpropagation (32:10 - 1:09:02)
    Once basit bir ifade, sonra tek bir noron uzerinde
    gradyanlari ELLE dolduruyoruz - chain rule'u icsellestirmek icin.
    """
    print("=" * 60)
    print("GOREV 2a: Manuel backprop - basit ifade")
    print("=" * 60)

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f

    print("L.data:", L.data)

    # 1. Taban durumu: dL/dL = 1
    L.grad = 1.0

    # 2. L = d * f  ->  dL/dd = f, dL/df = d
    d.grad = f.data * L.grad
    f.grad = d.data * L.grad

    # 3. d = e + c  ->  toplama lokal turevi 1, gradyan oldugu gibi dagilir
    c.grad = 1.0 * d.grad
    e.grad = 1.0 * d.grad

    # 4. e = a * b  ->  dL/da = b, dL/db = a
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    print("a.grad:", a.grad, "  b.grad:", b.grad, "  c.grad:", c.grad)
    print("e.grad:", e.grad, "  d.grad:", d.grad, "  f.grad:", f.grad)
    print()

    print("=" * 60)
    print("GOREV 2b: Manuel backprop - tek noron (tanh dahil)")
    print("=" * 60)

    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b2 = Value(6.8813735870195432)  # elle takip edebilmek icin "guzel" secilmis sayi

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    n = x1w1 + x2w2 + b2
    o = n.tanh()

    print("o.data:", o.data)

    # 1. Taban durumu: do/do = 1
    o.grad = 1.0

    # 2. o = tanh(n)  ->  do/dn = 1 - tanh(n)^2 = 1 - o.data^2
    n.grad = (1 - o.data ** 2) * o.grad

    # 3. n = x1w1 + x2w2 + b  ->  toplama lokal turevi 1, gradyan dagilir
    x1w1.grad = 1.0 * n.grad
    x2w2.grad = 1.0 * n.grad
    b2.grad = 1.0 * n.grad

    # 4. x1w1 = x1 * w1  ->  carpma: karsi tarafin data'si ile carp
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad

    # 5. x2w2 = x2 * w2
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad

    print("x1.grad:", x1.grad, "(beklenen: -1.5)")
    print("w1.grad:", w1.grad, "(beklenen: 1.0)")
    print("x2.grad:", x2.grad, "(beklenen: 0.5)")
    print("w2.grad:", w2.grad, "(beklenen: 0.0)")
    print()


def gorev_3():
    """
    GOREV 3: backward() metodu (1:09:02 - 1:27:05)
    Elle yaptigimizi otomatiklestiriyoruz: her Value kendi _backward
    fonksiyonunu saklar, backward() topolojik siralamayla hepsini
    dogru sirada cagirir. Ayni degisken birden fazla kullanilirsa
    gradyanlar TOPLANIR (+=), uzerine yazilmaz.
    """
    print("=" * 60)
    print("GOREV 3: backward() metodu (otomatik)")
    print("=" * 60)

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
    print("x1.grad:", x1.grad, "(beklenen: -1.5)")
    print("w1.grad:", w1.grad, "(beklenen: 1.0)")
    print("x2.grad:", x2.grad, "(beklenen: 0.5)")
    print("w2.grad:", w2.grad, "(beklenen: 0.0)")

    # Bug testi: ayni degiskenin birden fazla kullanimi
    a = Value(3.0)
    b2 = a + a
    b2.backward()
    print("\na.grad (a+a):", a.grad, "(beklenen: 2.0 - += sayesinde toplaniyor)")
    print()


def gorev_4():
    """
    GOREV 4: tanh'i parcala + uc yontemle dogrulama (1:27:05 - 1:43:55)
    tanh'i exp, +, -, / ile parcalayip ayni gradyanlari aldigimizi
    gosteriyoruz. Sonra backward(), numerical derivative ve PyTorch'u
    karsilastirip uc yontemin de eslestigini dogruluyoruz.
    """
    print("=" * 60)
    print("GOREV 4a: tanh'i parcala (exp, +, -, /)")
    print("=" * 60)

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
        # tanh'i atomik parcalara bol: (e^2x - 1) / (e^2x + 1)
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
    print("\n[Parcalanmis tanh: exp, +, -, /]")
    print(f"o.data = {ob.data}")
    print(f"x1.grad={x1b.grad}, w1.grad={w1b.grad}, x2.grad={x2b.grad}, w2.grad={w2b.grad}")
    print()

    print("=" * 60)
    print("GOREV 4b: backward() vs numerical derivative vs PyTorch")
    print("=" * 60)

    def loss_from_w1(w1_value):
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(w1_value)
        w2 = Value(1.0)
        b = Value(6.8813735870195432)
        n = x1 * w1 + x2 * w2 + b
        return n.tanh().data

    # 1) backward()
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()
    grad_backward = w1.grad

    # 2) Numerical derivative (gecen haftaki yontem)
    h = 0.0001
    loss1 = loss_from_w1(-3.0)
    loss2 = loss_from_w1(-3.0 + h)
    grad_numerical = (loss2 - loss1) / h

    # 3) PyTorch autograd
    try:
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
    except ImportError:
        grad_pytorch = None
        print("(PyTorch kurulu degil, bu kismi atliyorum)")

    print(f"\nw1'e gore gradyan (dL/dw1), uc yontemle:")
    print(f"  1) backward()           : {grad_backward:.6f}")
    print(f"  2) numerical derivative  : {grad_numerical:.6f}")
    if grad_pytorch is not None:
        print(f"  3) PyTorch autograd      : {grad_pytorch:.6f}")
    print()


def gorev_5():
    """
    GOREV 5: Neuron, Layer, MLP ve egitim dongusu (1:43:55 - 2:14:03)
    parameters() ile tum agirliklari topluyoruz, videodaki kucuk veri
    setiyle egitiyoruz. Her adimda gradyanlari sifirlamayi UNUTMUYORUZ
    (videodaki meshur bug).
    """
    print("=" * 60)
    print("GOREV 5: Neuron, Layer, MLP ve egitim dongusu")
    print("=" * 60)

    import random
    random.seed(42)  # tekrarlanabilir sonuclar icin

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # istenen ciktilar

    n = MLP(3, [4, 4, 1])  # 3 girdi, iki adet 4 noronlu gizli katman, 1 cikti
    print("Toplam parametre sayisi:", len(n.parameters()))

    ypred = [n(x) for x in xs]
    print("Egitim oncesi tahminler:", [round(p.data, 4) for p in ypred])
    print()

    for k in range(20):
        # --- forward pass ---
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # --- gradyanlari sifirla (videodaki meshur bug burada onleniyor!) ---
        # _backward fonksiyonlari grad'i += ile biriktiriyor, bu yuzden
        # her adimda once sifirlamazsak, bir onceki adimin gradyani
        # bir sonrakine eklenir ve egitim tamamen bozulur.
        for p in n.parameters():
            p.grad = 0.0

        # --- backward pass ---
        loss.backward()

        # --- update: her parametreyi gradyaninin TERSI yonunde kaydir ---
        learning_rate = 0.1
        for p in n.parameters():
            p.data += -learning_rate * p.grad

        print(f"adim {k:2d}  loss = {loss.data:.6f}")

    ypred = [n(x) for x in xs]
    print("\nEgitim sonrasi tahminler:", [round(p.data, 4) for p in ypred])
    print("Hedefler                :", ys)


if __name__ == "__main__":
    gorev_1()
    gorev_2()
    gorev_3()
    gorev_4()
    gorev_5()
