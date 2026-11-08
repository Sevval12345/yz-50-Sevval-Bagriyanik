# Görev 1: Veriyi GPT-2 tokenizer'ıyla token'la ve shard (parça) dosyalarına böl.

import argparse
import os
import time
import numpy as np

EOT = 50256  # GPT-2'nin <|endoftext|> token'ı: dokümanların başına konur

def fineweb_dokumanlari():
    """FineWeb-Edu (sample-10BT) dokümanlarını indirmeden, akıtarak (streaming) verir."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for doc in ds:
        yield doc["text"]

def dosya_dokumani(yol):
    """Tek bir metin dosyasını tek doküman olarak verir."""
    with open(yol, "r", encoding="utf-8") as f:
        yield f.read()

def token_uret(dokumanlar, enc):
    """Her doküman: [<|endoftext|>] + metnin token'ları, uint16 dizisi olarak."""
    for metin in dokumanlar:
        tok = [EOT] + enc.encode_ordinary(metin)
        arr = np.array(tok)
        assert (0 <= arr).all() and (arr < 2 ** 16).all(), "token uint16'ya sığmıyor"
        yield arr.astype(np.uint16)

def shardlari_yaz(token_dizileri, cikti_dir, shard_boyutu, toplam, onek="fineweb"):
    """Gelen token dizilerini shard_boyutu'luk parçalara bölüp .npy olarak yazar.
    İlk shard 'val', kalanlar 'train'. toplam token'a gelince durur."""
    os.makedirs(cikti_dir, exist_ok=True)
    buf = np.empty(shard_boyutu, dtype=np.uint16)
    n, shard_i, gorulen = 0, 0, 0
    yazilanlar = []
    t0 = time.time()

    def kaydet(parca):
        nonlocal shard_i
        ad = f"{onek}_{'val' if shard_i == 0 else 'train'}_{shard_i:03d}.npy"
        yol = os.path.join(cikti_dir, ad)
        np.save(yol, parca)
        yazilanlar.append((yol, len(parca)))
        print(f"  yazıldı: {ad}  ({len(parca):,} token, {time.time() - t0:.0f} sn)", flush=True)
        shard_i += 1

    for dizi in token_dizileri:
        while len(dizi) > 0 and gorulen < toplam:
            yer = min(shard_boyutu - n, toplam - gorulen)
            al = dizi[:yer]
            buf[n:n + len(al)] = al
            n += len(al)
            gorulen += len(al)
            dizi = dizi[len(al):]
            if n == shard_boyutu:
                kaydet(buf)
                n = 0
        if gorulen >= toplam:
            break
    if n > 0:
        kaydet(buf[:n].copy())
    return yazilanlar

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", choices=["fineweb", "dosya"], default="fineweb")
    ap.add_argument("--yol", default=None, help="--kaynak dosya için metin dosyası")
    ap.add_argument("--toplam", type=lambda s: int(s.replace("_", "")), default=25_000_000, help="toplam token")
    ap.add_argument("--shard", type=lambda s: int(s.replace("_", "")), default=5_000_000, help="shard başına token")
    ap.add_argument("--cikti", default="fineweb_shards")
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    HERE = os.path.dirname(os.path.abspath(__file__))
    cikti = os.path.join(HERE, args.cikti)

    if args.kaynak == "fineweb":
        dokumanlar = fineweb_dokumanlari()
    else:
        assert args.yol, "--kaynak dosya için --yol ver"
        dokumanlar = dosya_dokumani(args.yol)

    print(f"hedef: {args.toplam:,} token, shard boyutu {args.shard:,} -> {cikti}")
    yazilan = shardlari_yaz(token_uret(dokumanlar, enc), cikti, args.shard, args.toplam)
    toplam = sum(n for _, n in yazilan)
    boyut_mb = sum(os.path.getsize(y) for y, _ in yazilan) / 1e6
    print(f"\nbitti: {len(yazilan)} shard, {toplam:,} token, diskte {boyut_mb:.0f} MB "
          f"(token başına {boyut_mb * 1e6 / toplam:.2f} byte)")
    print(f"val: {yazilan[0][1]:,} token | train: {toplam - yazilan[0][1]:,} token", flush=True)
    os._exit(0)   # arka plandaki indirme parçacıklarını beklemeden hemen çık