from neuron import neuron_forward


def layer_forward(inputs, weights, biases):
    """
    Bir katmanın forward pass'i: her nöron ayni girdileri alır ama
    kendi ağırlıkları ve bias'ıyla işler. neuron_forward'i her nöron
    icin tekrar tekrar cagirarak katmana genisletiyoruz.
    """
    outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        output = neuron_forward(inputs, neuron_weights, neuron_bias)
        outputs.append(output)
    return outputs
