import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
from dataset import load_text, build_vocab, encode, split_data, get_batch
from model import AttentionLanguageModel

torch.manual_seed(1337)

batch_size = 32
block_size = 8
n_embd = 32
max_iters = 5000
eval_interval = 1000
eval_iters = 200
learning_rate = 1e-3
device = "cpu"

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out

if __name__ == "__main__":
    text = load_text(os.path.join(os.path.dirname(__file__), '..', 'input.txt'))
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = split_data(data, train_frac=0.9)

    # Gorev 4'teki ayni modeli egitiyoruz
    model = AttentionLanguageModel(vocab_size, n_embd, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Model egitiliyor (gorev 4 ile ayni ayarlar)")
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"  adim {it:5d}: train loss {losses['train']:.4f}   val loss {losses['val']:.4f}")
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Okunakli, gercek bir cumle secelim
    sentence = "First Ci"  # tam block_size=8 karakter, metinde gercekten geciyor
    assert len(sentence) == block_size
    idx = torch.tensor([encode(sentence, stoi)], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        model(idx)  # forward calistir ki Head.wei dolsun

    wei = model.sa_head.wei[0].numpy()  # (T, T)
    chars = list(sentence)

    # İsi haritasi
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(wei, cmap="viridis")
    ax.set_xticks(range(block_size))
    ax.set_yticks(range(block_size))
    ax.set_xticklabels(chars)
    ax.set_yticklabels(chars)
    ax.set_xlabel("Hangi karaktere bakiliyor (key pozisyonu)")
    ax.set_ylabel("Hangi karakter bakiyor (query pozisyonu)")
    ax.set_title(f"Attention agirliklari: '{sentence}'")
    for i in range(block_size):
        for j in range(block_size):
            ax.text(j, i, f"{wei[i, j]:.2f}", ha="center", va="center",
                     color="white" if wei[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="agirlik")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    print("\nİsi haritasi 'attention_heatmap.png' olarak kaydedildi.")

    print(f"\nCumle: {sentence!r}")
    print("Agirlik matrisi (satir=bakan, sutun=bakilan):")
    print(wei)