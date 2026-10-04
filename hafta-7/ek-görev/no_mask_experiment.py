import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
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

GOREV4_VAL_LOSS = 2.3972  # tril maskeli (dogru) halin val loss'u

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

    # Ek-5b: causal=False -> tril maskesi uygulanmiyor, her token kendisinden sonrakş tokenlara (kendi hedefi dahil) da bakabiliyor.
    model = AttentionLanguageModel(vocab_size, n_embd, block_size, causal=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Model egitiliyor (tril maskesi kaldirldi)")
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
    no_mask_val_loss = final_losses["val"]

    print("\n" + "=" * 55)
    print("KARSILASTIRMA: tril MASKELI (gorev 4) vs MASKESIZ (5b)")
    print("=" * 55)
    print(f"{'Versiyon':<30}{'Val Loss':<10}")
    print("-" * 40)
    print(f"{'tril maskeli (gorev 4, dogru)':<30}{GOREV4_VAL_LOSS:<10.4f}")
    print(f"{'tril maskesiz (gorev 5b)':<30}{no_mask_val_loss:<10.4f}")
    print(f"\nFark: {GOREV4_VAL_LOSS - no_mask_val_loss:+.4f}  (pozitifse maskesiz 'daha iyi' görünüyor demek)")

    print("\nUretilen metin ornegi (maskesiz model, 300 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=300)[0].tolist(), itos)
    print(generated)