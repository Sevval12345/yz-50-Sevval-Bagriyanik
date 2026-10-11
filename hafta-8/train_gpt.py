# Görev 3

import time
import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
from model import GPTLanguageModel

torch.manual_seed(1337)

# Konfigurasyon 
'''
CPU'da egitiyoruz (GPU yok). Karpathy'nin GPU konfigurasyonu
(n_embd=384, n_layer=6, block_size=256, yaklasik 10M parametre) burada cok
yavas olurdu. Once zamanlama testi yaptim:
- n_embd=128, n_head=4, n_layer=4, block_size=64 -> 816K param, yaklasik 16 dakika
- n_embd=64,  n_head=4, n_layer=3, block_size=32 -> 160K param, yasklasik 3 daikaka
Ikincisini sectim: hala onceki haftaya gore anlamli bir buyume
(block_size 8->32, n_embd 32->64, tek Block->3 Block) ama CPU'da
makul surede (yaklasik 3 dk) bitiyor.
'''

n_embd = 64
n_head = 4
n_layer = 3
block_size = 32
dropout = 0.2
batch_size = 32
max_iters = 5000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Konfigurasyon: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, block_size={block_size}, dropout={dropout}")
    print(f"Toplam parametre sayisi: {n_params}")
    print(f"Cihaz: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nEgitim basliyor")
    t0 = time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            print(f"  adim {it:5d} ({elapsed:6.1f}s): train loss {losses['train']:.4f}   val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - t0
    final_losses = estimate_loss(model, train_data, val_data)

    print("\n" + "=" * 50)
    print("SONUC")
    print("=" * 50)
    print(f"Parametre sayisi: {n_params}")
    print(f"Nihai val loss:   {final_losses['val']:.4f}")
    print(f"Egitim suresi:    {total_time:.1f} saniye ({total_time/60:.1f} dakika)")

    print("\nUretilen metin ornegi (500 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos)
    print(generated)