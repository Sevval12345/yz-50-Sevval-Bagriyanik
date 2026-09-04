def mse_loss(outputs, targets):
    """
    Mean Squared Error: ciktilar ile hedefler arasindaki farkin
    karesini alip ortalamasini hesaplar. Modelin ne kadar "kotu"
    oldugunu tek bir sayiyla olcer.
    """
    total = 0
    for output, target in zip(outputs, targets):
        total += (output - target) ** 2
    return total / len(outputs)
