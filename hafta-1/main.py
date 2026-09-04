from neuron import neuron_forward
from layer import layer_forward
from loss import mse_loss
from gradient import numerical_gradient


def gorev_1():
    """
    GOREV 1: Tek noron forward pass, kutuphanesiz.
    """
    print("=" * 60)
    print("GOREV 1: Tek noron forward pass")
    print("=" * 60)

    inputs = [0.5, 0.8, 0.2]
    weights = [0.4, -0.6, 0.9]
    bias = 0.1

    output = neuron_forward(inputs, weights, bias)
    print("Aktivasyon:", output)
    print()


def gorev_2():
    """
    GOREV 2: Birden fazla noron - katman (layer) forward pass.
    """
    print("=" * 60)
    print("GOREV 2: Katman (layer) forward pass")
    print("=" * 60)

    inputs = [0.5, 0.8, 0.2]

    weights = [
        [0.4, -0.6, 0.9],   # noron 1
        [0.1, 0.7, -0.3],   # noron 2
    ]

    biases = [0.1, -0.2]

    outputs = layer_forward(inputs, weights, biases)
    print("Katman ciktilari:", outputs)
    print()


def gorev_3():
    """
    GOREV 3: Basit loss fonksiyonu (MSE).
    """
    print("=" * 60)
    print("GOREV 3: Loss fonksiyonu (MSE)")
    print("=" * 60)

    outputs = [0.5, 0.5866]
    targets = [0, 1]

    loss = mse_loss(outputs, targets)
    print("Loss:", loss)
    print()


def gorev_4():
    """
    GOREV 4: Bir agirligi manuel degistirip loss'un nasil degistigini
    gozlemleme ve loss egrisini cizme.
    """
    print("=" * 60)
    print("GOREV 4: Parametre tarama + loss egrisi")
    print("=" * 60)

    inputs = [0.5, 0.8, 0.2]
    targets = [0, 1]
    biases = [0.1, -0.2]

    base_weights = [
        [0.4, -0.6, 0.9],   # noron 1 (sabit)
        [0.1, 0.7, -0.3],   # noron 2 (w0'i degistirecegiz)
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
    print(f"En dusuk loss: {best_loss:.4f}  (w = {best_w:.2f})")

    # Grafik cizme (matplotlib gerekiyor, plotting icin kutuphane sart)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(w_values, loss_values, color="#4C72B0", linewidth=2)
        plt.scatter([best_w], [best_loss], color="red", zorder=5, label=f"min loss (w={best_w:.2f})")
        plt.xlabel("Agirlik degeri (noron 2, w0)")
        plt.ylabel("Loss (MSE)")
        plt.title("Bir Agirligin Loss Uzerindeki Etkisi")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        print("Grafik 'loss_curve.png' olarak kaydedildi.")
    except ImportError:
        print("(matplotlib kurulu degil, grafik cizilemedi)")
    print()


def gorev_5():
    """
    GOREV 5: Sayisal turev ile basit bir gradient descent dongusu.
    """
    print("=" * 60)
    print("GOREV 5: Sayisal turev ile gradient descent")
    print("=" * 60)

    inputs = [0.5, 0.8, 0.2]
    targets = [0, 1]
    biases = [0.1, -0.2]

    base_weights = [
        [0.4, -0.6, 0.9],   # noron 1 (sabit)
        [0.1, 0.7, -0.3],   # noron 2 (w0'i optimize edecegiz)
    ]

    def compute_loss_for_weight(w):
        weights = [
            base_weights[0][:],
            [w, base_weights[1][1], base_weights[1][2]],
        ]
        outputs = layer_forward(inputs, weights, biases)
        return mse_loss(outputs, targets)

    w = 0.1              # baslangic degeri
    learning_rate = 1.0
    steps = 50

    print(f"Baslangic: w = {w:.4f}, loss = {compute_loss_for_weight(w):.4f}")

    for i in range(steps):
        loss = compute_loss_for_weight(w)
        grad = numerical_gradient(compute_loss_for_weight, w)

        w = w - learning_rate * grad

        if i % 5 == 0 or i == steps - 1:
            print(f"Adim {i:2d}: w = {w:.4f}, loss = {loss:.6f}, gradient = {grad:.4f}")

    print(f"\nSon durum: w = {w:.4f}, loss = {compute_loss_for_weight(w):.6f}")


if __name__ == "__main__":
    gorev_1()
    gorev_2()
    gorev_3()
    gorev_4()
    gorev_5()
