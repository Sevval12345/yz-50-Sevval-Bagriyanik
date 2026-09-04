def sigmoid(z):
    """
    Sigmoid aktivasyon fonksiyonu - girdiyi 0 ile 1 arasina sikistirir.
    Kutuphanesiz, Euler sayisini elle tanimlayarak.
    """
    e = 2.718281828459045  # Euler sayisi (yaklasik)
    return 1 / (1 + e ** (-z))


def neuron_forward(inputs, weights, bias):
    """
    Tek bir noronun forward pass'i: girdileri agirliklarla carpip
    topluyor, bias ekliyor, sigmoid ile sikistiriyor.
    """
    # 1. Agirlikli toplami hesapla
    weighted_sum = 0
    for x, w in zip(inputs, weights):
        weighted_sum += x * w

    # 2. Bias ekle
    z = weighted_sum + bias

    # 3. Sigmoid uygula
    activation = sigmoid(z)

    return activation
