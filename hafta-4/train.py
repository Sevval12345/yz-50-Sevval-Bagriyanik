import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import forward_pass, forward_pass_bn

def overfit_single_minibatch(Xtr, Ytr, parameters, batch_size=32, steps=1000, lr=0.1):
    """
    Kontrol: TEK BIR minibatch'i overfit ediyoruz. Loss
    neredeyse sifira inmeli -> inmiyorsa kodda bir hata var demek.
    """
    C, W1, b1, W2, b2 = parameters
    Xb, Yb = Xtr[:batch_size], Ytr[:batch_size]

    for k in range(steps):
        logits = forward_pass(Xb, C, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -lr * p.grad

        if k % 200 == 0 or k == steps - 1:
            print(f"  adim {k:4d}  loss = {loss.item():.4f}")

    print(f"-> Loss sifira yaklasti mi? {'EVET' if loss.item() < 0.5 else 'HAYIR'}")
    return parameters

def search_learning_rate(Xtr, Ytr, parameters, batch_size=32, steps=1000, save_path="lr_search.png"):
    """
    Ustel araliktaki (10^-3 ile 10^0 arasi) learning rate degerlerini
    sirayla deneyip loss'u kaydediyoruz, en dusuk loss'u veren degeri
    seciyoruz.
    """
    lre = torch.linspace(-3, 0, steps)
    lrs = 10 ** lre

    lri_history, loss_history = [], []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        C, W1, b1, W2, b2 = parameters
        logits = forward_pass(Xb, C, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = lrs[i].item()
        for p in parameters:
            p.data += -lr * p.grad

        lri_history.append(lre[i].item())
        loss_history.append(loss.item())

    min_idx = loss_history.index(min(loss_history))
    best_lr = 10 ** lri_history[min_idx]
    print(f"En dusuk loss'u veren lr: {best_lr:.4f} (log10={lri_history[min_idx]:.2f})")

    plt.figure(figsize=(8, 5))
    plt.plot(lri_history, loss_history)
    plt.axvline(lri_history[min_idx], color="red", linestyle="--", label=f"secilen: lr={best_lr:.3f}")
    plt.xlabel("log10(learning rate)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Tarama")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Grafik '{save_path}' olarak kaydedildi.")

    return best_lr

def train(Xtr, Ytr, Xdev, Ydev, parameters, learning_rate, batch_size=32, steps=20000, eval_every=2000):
    """
    Butun veriyi minibatch'lerle egitiyoruz. Her eval_every adimda
    train ve dev loss'unu raporluyoruz.
    """
    C, W1, b1, W2, b2 = parameters

    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = forward_pass(Xb, C, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -learning_rate * p.grad

        if k % eval_every == 0 or k == steps - 1:
            with torch.no_grad():
                dev_logits = forward_pass(Xdev, C, W1, b1, W2, b2)
                dev_loss = F.cross_entropy(dev_logits, Ydev)
            print(f"  adim {k:5d}  train loss = {loss.item():.4f}   dev loss = {dev_loss.item():.4f}")

    return parameters

def train_bn(Xtr, Ytr, Xdev, Ydev, parameters, running_mean, running_std,
             learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000):
    """train()'in BatchNorm'lu versiyonu. forward_pass_bn kullanir."""
    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = forward_pass_bn(Xb, parameters, running_mean, running_std, training=True)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -learning_rate * p.grad

        if k % eval_every == 0 or k == steps - 1:
            with torch.no_grad():
                dev_logits = forward_pass_bn(Xdev, parameters, running_mean, running_std, training=False)
                dev_loss = F.cross_entropy(dev_logits, Ydev)
            print(f"  adim {k:5d}  train loss = {loss.item():.4f}   dev loss = {dev_loss.item():.4f}")

    return parameters, running_mean, running_std

def train_with_lr_decay(Xtr, Ytr, Xdev, Ydev, parameters,
                         steps_phase1=40000, lr_phase1=0.1,
                         steps_phase2=20000, lr_phase2=0.01,
                         batch_size=32, eval_every=10000):
    """
    Iki fazli egitim: once buyuk lr ile kaba ayar, sonra kucuk lr ile
    hassas ayar (fine-tuning) yapiyoruz. Minimum noktanin etrafinda "zipla-
    durma"yi onluyor.
    """
    C, W1, b1, W2, b2 = parameters
    total_steps = steps_phase1 + steps_phase2

    for k in range(total_steps):
        lr = lr_phase1 if k < steps_phase1 else lr_phase2

        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = forward_pass(Xb, C, W1, b1, W2, b2)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -lr * p.grad

        if k % eval_every == 0 or k == total_steps - 1:
            with torch.no_grad():
                dev_logits = forward_pass(Xdev, C, W1, b1, W2, b2)
                dev_loss = F.cross_entropy(dev_logits, Ydev)
            faz = 1 if k < steps_phase1 else 2
            print(f"  [faz {faz}] adim {k:6d}  lr={lr:<5}  train loss = {loss.item():.4f}   dev loss = {dev_loss.item():.4f}")

    return parameters