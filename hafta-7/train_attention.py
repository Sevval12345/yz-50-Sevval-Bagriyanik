import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
from model import AttentionLanguageModel

torch.manual_seed(1337)

#  Hiperparametreler 
# block_size, batch_size: gorev-1'deki bigram tabanla ayni tutuyoruz ki karsilastirma adil olsun, tek degisken attention'in eklenmis olmasi.
batch_size = 32
block_size = 8
n_embd = 32
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3  # bigram'dan dusuk: artik daha fazla katman/parametre var, buyuk lr'de kararsizlasabilir
device = "cuda" if torch.cuda.is_available() else "cpu"

BIGRAM_BASELINE_VAL_LOSS = 2.4762  # görev-1'den

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

    model = AttentionLanguageModel(vocab_size, n_embd, block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayisi: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nEgitim basliyor (Head'li model)")
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
    attention_val_loss = final_losses["val"]

    print("\n" + "=" * 50)
    print("KARSILASTIRMA (gorev 1 bigram tabanina karsi)")
    print("=" * 50)
    print(f"{'Model':<28}{'Val Loss':<10}")
    print("-" * 38)
    print(f"{'Bigram (gorev 1)':<28}{BIGRAM_BASELINE_VAL_LOSS:<10.4f}")
    print(f"{'Bigram + 1 attention head':<28}{attention_val_loss:<10.4f}")
    print(f"\nFark: {BIGRAM_BASELINE_VAL_LOSS - attention_val_loss:+.4f}  (pozitifse attention daha iyi)")

    print("\nUretilen metin ornegi (300 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=300)[0].tolist(), itos)
    print(generated)