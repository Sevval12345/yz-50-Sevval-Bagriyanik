# Ek Görev: Türkçe metni GPT-2 tokenizer'ıyla token'la, train/val shard'larına böl.

import argparse
import os
import numpy as np
import tiktoken

EOT = 50256
ap = argparse.ArgumentParser()
ap.add_argument("--yol", required=True)
ap.add_argument("--val-oran", type=float, default=0.1)
ap.add_argument("--cikti", default="turkce_shards")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
cikti = os.path.join(HERE, args.cikti)
os.makedirs(cikti, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
with open(args.yol, "r", encoding="utf-8") as f:
    metin = f.read()
tok = np.array([EOT] + enc.encode_ordinary(metin), dtype=np.uint16)

n_val = int(len(tok) * args.val_oran)
val, train = tok[-n_val:], tok[:-n_val]
np.save(os.path.join(cikti, "turkce_val_000.npy"), val)
np.save(os.path.join(cikti, "turkce_train_001.npy"), train)

B, T, TOTAL = 4, 256, 4096
print(f"metin: {len(metin):,} karakter, {len(metin.encode('utf-8')):,} byte")
print(f"token: {len(tok):,} (token başına {len(metin.encode('utf-8')) / len(tok):.2f} byte)")
print(f"train: {len(train):,} | val: {len(val):,}")
print(f"1 epoch = {len(train) // TOTAL} adım (adım başına {TOTAL} token)")
print(f"önerilen --val-steps: {max(1, min(20, (len(val) - 1) // (B * T)))}")
assert len(val) > B * T + 1 and len(train) > B * T + 1, "metin çok kısa"

"""
- HellaSwag İngilizce olduğu için Türkçe modeli nasıl ölçerdin? -

HellaSwag İngilizce olduğu için Türkçe modelde doğrudan kullanamam. Ben iki şey ölçtüm.

1. Tutulmuş Türkçe metinde loss. Val loss 3,56, bits-per-byte 2,42. Byte bazlı ölçmemin
sebebi tokenizer’ın Türkçe’yi İngilizce’den çok daha fazla parçaya bölmesi
(token başına 2,12 byte). Ham loss farklı tokenizer’lar arasında adil kıyas vermez.

2) Kendi mini testim. HellaSwag’in mantığını kopyaladım: bağlam ve 4 devam, en düşük
loss’lu olanı seçiyor. Doğru devam tutulmuş metinden, yanlışlar metnin başka yerlerinden.
Model %35,6 buldu, rastgele %25. Test kolay olduğu için bu sonuç da aslında düşük, model
henüz bağlamı pek kullanmıyor.

Gerçek bir ölçüm için Türkçe’ye çevrilmiş HellaSwag ya da Türkçe içeren bir benchmark
(XCOPA gibi) kullanırdım, ama bunların varlığını ve kalitesini ben kontrol etmedim.
"""