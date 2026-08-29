def sigmoid(z):
    e = 2.718281828459045  # Yaklaşık Euler sayısı (burada ai'dan yardım aldım) 
    return 1 / (1 + e ** (-z))

def neuron_forward(inputs, weights, bias):
    # Önce ağırlıklı toplamı hesaplayalım
    weighted_sum = 0
    for x, w in zip(inputs, weights):
        weighted_sum += x * w

    # Bias ekleyelim
    z = weighted_sum + bias

    # Sigmoid uygulayalım
    activation = sigmoid(z)

    return activation

# Test
inputs = [0.5, 0.8, 0.2]
weights = [0.4, -0.6, 0.9]
bias = 0.1

output = neuron_forward(inputs, weights, bias)
print("Aktivasyon:", output)
