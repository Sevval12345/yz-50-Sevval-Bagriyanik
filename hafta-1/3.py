def mse_loss(outputs, targets):
    total = 0
    for output, target in zip(outputs, targets):
        total += (output - target) ** 2
    return total / len(outputs)

# Test
outputs = [0.5, 0.5866]
targets = [0, 1]

loss = mse_loss(outputs, targets)
print("Loss:", loss)
