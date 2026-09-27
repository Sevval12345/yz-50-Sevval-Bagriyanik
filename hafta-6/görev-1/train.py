import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(Xtr, Ytr, Xdev, Ydev, model, learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000):
    """
    Egitim dongusu katmanlarin isimlerini bilmiyor. Sadece model(Xb) ile
    forward pass yapıyor, model.parameters() ile guncellenecek tensorleri topluyor.
    model.train()/model.eval() ile BatchNorm1d'nin calisma modunu yonetiyor.

    Modelden cikan logits 3 boyutlu ise (N, 1, vocab_size), F.cross_entropy
    2 boyutlu (N, vocab_size) tensör bekledigi icin squeeze(1) uygulaniyor.
    """
    lossi = []

    model.train()
    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Forward pass
        logits = model(Xb)
        if logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = F.cross_entropy(logits, Yb)

        # Backward pass & update
        for p in model.parameters():
            p.grad = None
        loss.backward()
        for p in model.parameters():
            p.data += -learning_rate * p.grad

        lossi.append(loss.log10().item())

        # Degerlendirme adimi
        if k % eval_every == 0 or k == steps - 1:
            model.eval()
            with torch.no_grad():
                dev_logits = model(Xdev)
                if dev_logits.ndim == 3 and dev_logits.shape[1] == 1:
                    dev_logits = dev_logits.squeeze(1)
                dev_loss = F.cross_entropy(dev_logits, Ydev)
            model.train()
            print(f"  adim {k:5d}  train loss = {loss.item():.4f}   dev loss = {dev_loss.item():.4f}")

    model.eval()
    return model, lossi

def plot_loss_curve(lossi, group_size=1000, save_path="loss_curve.png"):
    """
    Loss egrisini duzeltiyoruz. Ham (her adim) loss cok
    gurultulu oluyor. lossi listesini group_size'lik gruplara bolup,
    her grubun ortalamasini aliyoruz -> videodaki "lossi.view(-1, 1000).mean(1)" 
    """
    n = len(lossi)
    n_full_groups = n // group_size
    trimmed = lossi[:n_full_groups * group_size]  # tam bolunemeyen kalani atiyoruz

    grouped = torch.tensor(trimmed).view(-1, group_size).mean(1)

    plt.figure(figsize=(8, 5))
    plt.plot(grouped)
    plt.xlabel(f"Adim (x{group_size})")
    plt.ylabel("log10(loss) - ortalama")
    plt.title(f"Loss Egrisi (her {group_size} adimin ortalamasi)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss egrisi '{save_path}' olarak kaydedildi.")

if __name__ == "__main__":
    from dataset import read_words, build_vocab, build_dataset, split_words
    from model import build_model

    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 3, 10, 200
    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=block_size)

    model = build_model(vocab_size, embedding_dim, block_size, n_hidden)

    print("\nModelin katmanları ve çıktı şekilleri (bir batch ile):")
    Xb, Yb = Xtr[:4], Ytr[:4]
    x = Xb
    for layer in model.layers:
        x = layer(x)
        print(f"  {layer.__class__.__name__:<12} çıktı şekli: {tuple(x.shape)}")
    print()

    model, lossi = train(Xtr, Ytr, Xdev, Ydev, model,
                          learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000)

    with torch.no_grad():
        final_logits = model(Xdev)
        final_dev_loss = F.cross_entropy(final_logits, Ydev).item()
    print(f"\nNihai dev loss: {final_dev_loss:.4f}")

    plot_loss_curve(lossi, group_size=1000)