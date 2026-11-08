# Görev 1: shard dosyalarından okuyan data loader.
# Tek başına çalıştırınca doğrulama testleri yapar.

import os
import sys
import numpy as np
import torch

def shard_yukle(yol):
    """uint16 shard'ı belleğe alır. Batch'e girerken int64'e çevrilir (embedding long ister)."""
    return np.load(yol)

class ShardedDataLoader:
    def __init__(self, shard_dir, split, B, T):
        assert split in ("train", "val")
        self.B, self.T = B, T
        self.shards = sorted(os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
                             if f.endswith(".npy") and f"_{split}_" in f)
        assert len(self.shards) > 0, f"{shard_dir} içinde {split} shard'ı yok"
        self.split = split
        self.reset()

    def reset(self):
        self.shard_i = 0
        self.tokens = shard_yukle(self.shards[0])
        self.pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos + B * T + 1].astype(np.int64)   # +1: son token'ın hedefi
        x = torch.from_numpy(buf[:-1].reshape(B, T))
        y = torch.from_numpy(buf[1:].reshape(B, T))
        self.pos += B * T
        if self.pos + B * T + 1 > len(self.tokens):         # sonraki batch bu shard'a sığmıyor
            self.shard_i = (self.shard_i + 1) % len(self.shards)
            self.tokens = shard_yukle(self.shards[self.shard_i])
            self.pos = 0
        return x, y

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    shard_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fineweb_shards")
    B, T = 4, 256

    tr = ShardedDataLoader(shard_dir, "train", B, T)
    va = ShardedDataLoader(shard_dir, "val", B, T)
    print(f"train shard: {len(tr.shards)} | val shard: {len(va.shards)}")
    for ad, ld in (("train", tr), ("val", va)):
        for y in ld.shards:
            print(f"  [{ad}] {os.path.basename(y)}: {len(shard_yukle(y)):,} token")
    ortak = {os.path.basename(a) for a in tr.shards} & {os.path.basename(a) for a in va.shards}
    print(f"train ve val aynı dosyayı paylaşıyor mu: {bool(ortak)}")

    x, y = tr.next_batch()
    print(f"\nx {tuple(x.shape)} {x.dtype} | y {tuple(y.shape)} {y.dtype}")
    print(f"y, x'in bir kaydırılmışı mı: {bool((y[:, :-1] == x[:, 1:]).all())}")
    print(f"token aralığı: {x.min().item()} .. {x.max().item()}  (vocab 50257)")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print("\nilk satırın başı:\n" + repr(enc.decode(x[0, :60].tolist())))
    except Exception as e:                                     # tiktoken yoksa atla
        print(f"(decode atlandı: {e})")

    # shard sınırını geçiyor muyuz?
    tr.reset()
    ilk = tr.shard_i
    adim = 0
    while tr.shard_i == ilk and adim < 10_000_000:
        tr.next_batch()
        adim += 1
    print(f"\nilk shard {adim} batch sürdü, sonra shard {tr.shard_i}'e geçildi (pos={tr.pos})")
    for _ in range(3):
        x, y = tr.next_batch()
    print("shard geçişinden sonra batch şekli:", tuple(x.shape))