# Görev 2

import torch
from dataset import load_text, build_vocab, encode, split_data, get_batch
from model import MultiHeadFeedForwardLanguageModel, ResidualLanguageModel, BlockLanguageModel

torch.manual_seed(1337)

batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
max_iters = 5000
eval_interval = 1000
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Onceki haftalardan - gorev 1'den sabit degerler
BIGRAM_VAL_LOSS = 2.4762
SINGLE_HEAD_VAL_LOSS = 2.3972
MULTIHEAD_VAL_LOSS = 2.2692

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

def train_model(model_cls, label, vocab_size, train_data, val_data):
    model = model_cls(vocab_size, n_embd, block_size, n_head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\n--- {label} egitiliyor")
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
    return final_losses["val"]

if __name__ == "__main__":
    text = load_text("input.txt")
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = split_data(data, train_frac=0.9)

    ff_val_loss = train_model(
        MultiHeadFeedForwardLanguageModel, "ADIM 1: multi-head + feedforward (residual/LN YOK)",
        vocab_size, train_data, val_data)

    residual_val_loss = train_model(
        ResidualLanguageModel, "ADIM 2: + residual baglanti (LN hala YOK)",
        vocab_size, train_data, val_data)

    layernorm_val_loss = train_model(
        BlockLanguageModel, "ADIM 3: + LayerNorm (tam Transformer Blogu)",
        vocab_size, train_data, val_data)

    print("\n" + "=" * 55)
    print("TEK TABLO: bigram -> tam transformer blogu")
    print("=" * 55)
    print(f"{'Asama':<35}{'Val Loss':<10}")
    print("-" * 45)
    print(f"{'Bigram':<35}{BIGRAM_VAL_LOSS:<10.4f}")
    print(f"{'Tek head':<35}{SINGLE_HEAD_VAL_LOSS:<10.4f}")
    print(f"{'Multi-head':<35}{MULTIHEAD_VAL_LOSS:<10.4f}")
    print(f"{'+ feedforward':<35}{ff_val_loss:<10.4f}")
    print(f"{'+ residual':<35}{residual_val_loss:<10.4f}")
    print(f"{'+ LayerNorm':<35}{layernorm_val_loss:<10.4f}")