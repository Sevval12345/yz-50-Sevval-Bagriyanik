import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
from model import BigramLanguageModel

torch.manual_seed(1337)

# Hiperparametreler 
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2  # bigram cok basit bir model oldugu icin yuksek lr'de bile stabil
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    Tek bir batch'in loss'u gurultulu olur (hafta 4 ve 6'daki loss
    grafiginde gordugumuz zikzak gibi olur yanş). Bunun yerine eval_iters kadar
    batch'in ortalamasini aliyoruz. Hem train hem val icin, boylece
    print ettigimiz sayi dahs guvenilir oluyor. model.eval()/train()
    burada bigram icin islevsiz (dropout/batchnorm yok) ama alisilan
    torch.nn deseni bu, ileride attention'li modelde gerekecek.
    """
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

    print(f"Metin uzunlugu: {len(text)} karakter")
    print(f"Alfabe boyutu (vocab_size): {vocab_size}")
    print(f"Train: {len(train_data)} karakter, Val: {len(val_data)} karakter")

    model = BigramLanguageModel(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayisi: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nEgitim basliyor")
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
    print(f"\nNihai val loss (BIGRAM TABAN): {final_losses['val']:.4f}")

    print("\nUretilen metin ornegi (egitilmemis baglamdan, 300 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=300)[0].tolist(), itos)
    print(generated)