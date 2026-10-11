# Görev 4: aynı GPT'yi Türkçe ile eğit.

import sys
import time
import torch
from dataset import load_text, build_vocab, encode, decode, split_data, get_batch
from model import GPTLanguageModel

torch.manual_seed(1337)

# ---- AYNI konfigurasyon, gorev 3'teki Ingilizce modelle BIREBIR ayni ----
# Tek degisken: veri dili. Boylece "ayni model, farkli dil" karsilastirmasi
# adil oluyor - mimari/hiperparametre farki karismiyor.
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

ENGLISH_VAL_LOSS = 2.0136  # gorev 3'ten (Tiny Shakespeare)
ENGLISH_VOCAB_SIZE = 65


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
    # Korpus dosyasi komut satirindan verilebilir:
    #   python train_turkish_gpt.py                    -> turkce_metin.txt  (compile_turkish_corpus.py, v1)
    #   python train_turkish_gpt.py turkce_metin2.txt  -> v2 korpusu
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "turkce_metin2.txt"
    print(f"Korpus dosyasi: {corpus_path}")
    text = load_text(corpus_path)
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = split_data(data, train_frac=0.9)

    print(f"Turkce metin uzunlugu: {len(text)} karakter")
    print(f"Turkce vocab_size: {vocab_size}  (Ingilizce (Shakespeare): {ENGLISH_VOCAB_SIZE})")
    print(f"Fark: {vocab_size - ENGLISH_VOCAB_SIZE:+d} karakter")

    model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nToplam parametre sayisi: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nEgitim basliyor...")
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
    turkish_val_loss = final_losses["val"]

    print("\n" + "=" * 55)
    print("SONUC")
    print("=" * 55)
    print(f"Turkce vocab_size:  {vocab_size}  (Ingilizce: {ENGLISH_VOCAB_SIZE}, fark: {vocab_size - ENGLISH_VOCAB_SIZE:+d})")
    print(f"Turkce val loss:    {turkish_val_loss:.4f}")
    print(f"Ingilizce val loss: {ENGLISH_VAL_LOSS:.4f}  (gorev 3'ten, referans - DOGRUDAN KARSILASTIRILAMAZ, farkli vocab_size)")
    print(f"Egitim suresi:      {total_time:.1f} saniye ({total_time/60:.1f} dakika)")

    print("\nUretilen metin ornegi (500 karakter):\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos)
    print(generated)