def sigmoid(z):
    e = 2.718281828459045  # Euler 
    return 1 / (1 + e ** (-z))

def neuron_forward(inputs, weights, bias):
    # Ağırlıklı toplamı hesaplayalım
    weighted_sum = 0
    for x, w in zip(inputs, weights):
        weighted_sum += x * w

    # Bias ekleyelim
    z = weighted_sum + bias

    # Sigmoid uygulayalım
    activation = sigmoid(z)

    return activation

def layer_forward(inputs, weights, biases):
    outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        output = neuron_forward(inputs, neuron_weights, neuron_bias)
        outputs.append(output)
    return outputs

# Test
inputs = [0.5, 0.8, 0.2]
weights = [
    [0.4, -0.6, 0.9],   # nöron 1
    [0.1, 0.7, -0.3],   # nöron 2
]
biases = [0.1, -0.2]

outputs = layer_forward(inputs, weights, biases)
print("Katman çıktıları:", outputs)
