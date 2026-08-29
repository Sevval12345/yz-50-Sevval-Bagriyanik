def sigmoid(z):
    e = 2.718281828459045 
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

base_weights = [
    [0.4, -0.6, 0.9],   # nöron 1 (sabit)
    [0.1, 0.7, -0.3],   # nöron 2 (w0'ı optimize edeceğiz)
]

def compute_loss_for_weight(w):
    # nöron 2'nin ilk ağırlığını w yapıp, gerisini sabit tutalım
    weights = [
        base_weights[0][:],
        [w, base_weights[1][1], base_weights[1][2]],
    ]
    outputs = layer_forward(inputs, weights, biases)
    return mse_loss(outputs, targets)

def numerical_gradient(w):
    h = 0.0001
    loss1 = compute_loss_for_weight(w)
    loss2 = compute_loss_for_weight(w + h)
    return (loss2 - loss1) / h

# Gradient Descent Döngüsü
w = 0.1              # başlangıç değerimiz
learning_rate = 1.0
steps = 50

print(f"Başlangıç: w = {w:.4f}, loss = {compute_loss_for_weight(w):.4f}")

for i in range(steps):
    loss = compute_loss_for_weight(w)
    grad = numerical_gradient(w)

    w = w - learning_rate * grad

    if i % 5 == 0 or i == steps - 1:
        print(f"Adım {i:2d}: w = {w:.4f}, loss = {loss:.6f}, gradient = {grad:.4f}")

print(f"\nSon durum: w = {w:.4f}, loss = {compute_loss_for_weight(w):.6f}")