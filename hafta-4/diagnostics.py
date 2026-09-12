import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import forward_pass
from train import train

def analyze_saturation(h, label):
    """
    Doyma analizi: |h| > 0.99 olan degerlerin orani (tanh'in duzlestigi,
    gradyanin neredeyse sifir oldugu bolge). Ayrica her noronu, tum
    ornekler icin doymus olup olmadigina (tamamen "ölü" mu) bakiyoruz.
    """
    saturated_fraction = (h.abs() > 0.99).float().mean().item()
    fully_dead_neurons = (h.abs() > 0.99).all(dim=0).sum().item()
    print(f"[{label}] |h| > 0.99 olan degerlerin orani: {saturated_fraction:.2%}")
    print(f"[{label}] Tamamen 'ölü' noron sayisi: {fully_dead_neurons} / {h.shape[1]}")
    return saturated_fraction, fully_dead_neurons

def plot_histogram(values, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(values.detach().flatten().numpy(), bins=50)
    plt.title(title)
    plt.xlabel("Deger")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Histogram '{save_path}' olarak kaydedildi.")

def visualize_embeddings(C, itos, save_path="embeddings_2d.png"):
    """
    embedding_dim=2 oldugunda, her karakterin embedding'ini dogrudan
    bir sacilim grafiginde cizebiliyoruz. Birbirine yakin duran
    harfler, modelin onlari "benzer davranan" karakterler olarak
    ogrendigini gosteriyor.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].detach(), C[:, 1].detach(), s=200, color="lightblue")
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", fontsize=12)
    plt.grid(alpha=0.3)
    plt.title("Karakter Embedding'leri (2 Boyut)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Embedding grafigi '{save_path}' olarak kaydedildi.")

def compare_capacities(Xtr, Ytr, Xdev, Ydev, vocab_size, block_size, configs, steps=15000, lr=0.1):
    """
    Farkli 'embedding_dim / n_hidden' kombinasyonlarini deneyip dev
    loss'un nasil degistigini karsilastiriyoruz.
    """
    from model import init_layers_kaiming

    results = []
    print(f"{'embedding_dim':<15}{'n_hidden':<12}{'dev loss':<10}")
    print("-" * 37)

    for embedding_dim, n_hidden in configs:
        parameters = init_layers_kaiming(vocab_size, embedding_dim, block_size, n_hidden)
        parameters = train(Xtr, Ytr, Xdev, Ydev, parameters,
                            learning_rate=lr, batch_size=32, steps=steps, eval_every=steps)
        C, W1, b1, W2, b2 = parameters
        with torch.no_grad():
            dev_logits = forward_pass(Xdev, C, W1, b1, W2, b2)
            dev_loss = F.cross_entropy(dev_logits, Ydev).item()
        results.append((embedding_dim, n_hidden, dev_loss))
        print(f"{embedding_dim:<15}{n_hidden:<12}{dev_loss:<10.4f}")

    return results