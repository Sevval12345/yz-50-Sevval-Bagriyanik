# Görev 1

import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
from model import MultiHeadLanguageModel

torch.manual_seed(1337)

# Hiperparametreler 
# n_embd ve block_size gecen haftayla (hafta 7) ayni, tek degisken n_head: tek buyuk head yerine 4 kucuk.
batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
max_iters = 5000
eval_interval = 1000
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

SINGLE_HEAD_VAL_LOSS = 2.3972  # gecen hafta (hafta 7-gorev 4) tek head sonucu

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
    text = load_text("input.txt")
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = split_data(data, train_frac=0.9)

    model = MultiHeadLanguageModel(vocab_size, n_embd, block_size, n_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_head={n_head}, her head_size={n_embd // n_head}")
    print(f"Toplam parametre sayisi: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nEgitim basliyor (multi-head)")
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"  adim {it:5d}: train loss {losses['train']:.4f}   val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_losses = estimate_loss(model, train_data, val_data)
    multihead_val_loss = final_losses["val"]

    print("\n" + "=" * 55)
    print("Karsilastirma: tek head (hafta 7) vs multi-head")
    print("=" * 55)
    print(f"{'Model':<30}{'Val Loss':<10}")
    print("-" * 40)
    print(f"{'Tek head (hafta 7)':<30}{SINGLE_HEAD_VAL_LOSS:<10.4f}")
    print(f"{'Multi-head (n_head=4)':<30}{multihead_val_loss:<10.4f}")
    print(f"\nFark: {SINGLE_HEAD_VAL_LOSS - multihead_val_loss:+.4f}  (pozitifse multi-head daha iyi)")

    print("\nUretilen metin ornegi (300 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=300)[0].tolist(), itos)
    print(generated)