import matplotlib.pyplot as plt

def sigmoid(z):
    e = 2.718281828459045  # Euler 
    return 1 / (1 + e ** (-z))

def neuron_forward(inputs, weights, bias):
    weighted_sum = 0
    for x, w in zip(inputs, weights):
        weighted_sum += x * w
    z = weighted_sum + bias
    return sigmoid(z)

def layer_forward(inputs, weights, biases):
    outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        output = neuron_forward(inputs, neuron_weights, neuron_bias)
        outputs.append(output)
    return outputs

def mse_loss(outputs, targets):
    total = 0
    for output, target in zip(outputs, targets):
        total += (output - target) ** 2
    return total / len(outputs)

# Sabit değerler
inputs = [0.5, 0.8, 0.2]
targets = [0, 1]
biases = [0.1, -0.2]

# Nöron 2'nin ilk ağırlığını (w0) -3 ile 3 arasında değiştirip
# loss'un nasıl değiştiğini gözlemleyeceğiz.
base_weights = [
    [0.4, -0.6, 0.9],   # nöron 1 (sabit)
    [0.1, 0.7, -0.3],   # nöron 2 (w0'ı değiştireceğiz)
]

w_values = []
loss_values = []

w = -3.0
step = 0.05
while w <= 3.0:
    weights = [
        base_weights[0][:],
        [w, base_weights[1][1], base_weights[1][2]],
    ]
    outputs = layer_forward(inputs, weights, biases)
    loss = mse_loss(outputs, targets)

    w_values.append(w)
    loss_values.append(loss)
    w += step

min_index = loss_values.index(min(loss_values))
best_w = w_values[min_index]
best_loss = loss_values[min_index]
print(f"En düşük loss: {best_loss:.4f}  (w = {best_w:.2f})")

# Grafik 
plt.plot(w_values, loss_values)
plt.scatter([best_w], [best_loss], color="red")
plt.xlabel("Ağırlık değeri (nöron 2, w0)")
plt.ylabel("Loss (MSE)")
plt.title("Bir Ağırlığın Loss Üzerindeki Etkisi")
plt.show()