import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import load_text, build_vocab, encode, split_data, get_batch
from model import Head

torch.manual_seed(1337)

# Veri ve kucuk bir batch 
text = load_text(os.path.join(os.path.dirname(__file__), '..', 'input.txt'))
stoi, itos, vocab_size = build_vocab(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)
train_data, val_data = split_data(data)

block_size = 8
n_embd = 32
head_size = 16
batch_size = 4

xb, yb = get_batch(train_data, block_size, batch_size)

# Token + pozisyon embedding 
# Görev 3: attention sirayi bilmiyor, o yuzden pozisyon bilgisini ayrica ekliyoruz. tok_emb: "bu hangi karakter". pos_emb: "bu kacinci sirada".
token_embedding_table = nn.Embedding(vocab_size, n_embd)
position_embedding_table = nn.Embedding(block_size, n_embd)

tok_emb = token_embedding_table(xb)                              # (B, T, n_embd)
pos_emb = position_embedding_table(torch.arange(block_size))     # (T, n_embd)
x = tok_emb + pos_emb                                             # broadcast -> (B, T, n_embd)

print(f"xb (girdi token indeksleri) sekli: {tuple(xb.shape)}")
print(f"x (tok_emb + pos_emb) sekli: {tuple(x.shape)}")

# Tek bir self-attention head 
head = Head(n_embd, head_size, block_size)
out = head(x)
print(f"head ciktisi (out) sekli: {tuple(out.shape)}")

# wei matrisini yazdir, bir satirini oku
print("\n" + "=" * 60)
print("ILK ORNEGIN AGIRLIK MATRISI (wei), 8x8:")
print("=" * 60)
print(head.wei[0])

print("\n" + "=" * 60)
print("SON SATIRI (pozisyon 7) OKUYALIM:")
print("bu token, gecmisteki (ve kendisi dahil) hangi tokenlara ne kadar bakiyor")
print("=" * 60)
first_example_chars = [itos[i.item()] for i in xb[0]]
print(f"ilk ornegin karakterleri: {first_example_chars}")
for t in range(block_size):
    bar = "#" * int(head.wei[0, -1, t].item() * 40)
    print(f"  pozisyon {t} ({first_example_chars[t]!r:>4}) -> agirlik {head.wei[0, -1, t].item():.4f}  {bar}")


# Görev 3: Ölcekleme (scaling) etkisi 
print("\n" + "=" * 60)
print("SKORLARI head_size'IN KAREKOKUNE NEDEN BOLUYORUZ?")
print("=" * 60)

B, T, C = x.shape
k = head.key(x)
q = head.query(x)
print(f"q varyansi: {q.var().item():.4f}")
print(f"k varyansi: {k.var().item():.4f}")

wei_raw = q @ k.transpose(-2, -1)                    # Ölceksiz
wei_scaled = wei_raw * head_size ** -0.5             # Ölcekli (gercekte head icinde yapilan)

print(f"\nwei (olceksiz) varyansi:  {wei_raw.var().item():.4f}   (~head_size={head_size}'e yakin olmali)")
print(f"wei (olcekli /sqrt({head_size})) varyansi: {wei_scaled.var().item():.4f}   (~1'e yakin olmali)")

tril = torch.tril(torch.ones(T, T))
wei_raw_masked = wei_raw.masked_fill(tril == 0, float('-inf'))
wei_scaled_masked = wei_scaled.masked_fill(tril == 0, float('-inf'))

softmax_raw = F.softmax(wei_raw_masked, dim=-1)
softmax_scaled = F.softmax(wei_scaled_masked, dim=-1)

print("\nSON SATIR (pozisyon 7), YAN YANA:")
print(f"{'Bolmeden softmax':<22}{'Bolerek softmax':<22}")
print("-" * 44)
for a, b in zip(softmax_raw[0, -1].tolist(), softmax_scaled[0, -1].tolist()):
    print(f"{a:<22.4f}{b:<22.4f}")

print(f"\nBolmeden softmax'in en buyuk degeri: {softmax_raw[0, -1].max().item():.4f}  (1'e cok yakinsa = tek tokene kilitlenmis)")
print(f"Bolerek softmax'in en buyuk degeri:  {softmax_scaled[0, -1].max().item():.4f}  (daha dagitilmis olmali)")