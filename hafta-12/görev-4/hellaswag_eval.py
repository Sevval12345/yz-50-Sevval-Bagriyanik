# Görev 4: HellaSwag değerlendirmesi.

# Veri dosyası otomatik inmezse: hellaswag_val.jsonl dosyasını https://github.com/rowanz/hellaswag (data klasörü) adresindrn indirip bu klasöre koy.

import argparse
import json
import os
import urllib.request
import tiktoken
import torch
import torch.nn.functional as F
from gpt2 import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
GERCEK_VOCAB = 50257

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000, help="kaç örnek değerlendirilsin")
ap.add_argument("--yol", default=os.path.join(HERE, "hellaswag_val.jsonl"))
ap.add_argument("--model", default=os.path.join(HERE, "..", "görev-2", "saat1", "model.pt"))
ap.add_argument("--vocab", type=int, default=50304, help="kendi modelinin vocab boyutu")
args = ap.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"cihaz: {device}")

enc = tiktoken.get_encoding("gpt2")

def veriyi_hazirla(yol):
    """Önce yerel jsonl'e bakar; yoksa Hugging Face'ten (Rowan/hellaswag, validation) çeker ve jsonl olarak kaydeder."""
    if not os.path.exists(yol):
        from datasets import load_dataset
        print("indiriliyor: Hugging Face Rowan/hellaswag (validation)")
        ds = load_dataset("Rowan/hellaswag", split="validation")
        with open(yol, "w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps({"ctx": r["ctx"], "endings": r["endings"], "label": r["label"]},
                                   ensure_ascii=False) + "\n")
        print(f"{len(ds)} örnek kaydedildi: {yol}")
    with open(yol, "r", encoding="utf-8") as f:
        return [json.loads(satir) for satir in f if satir.strip()]

def ornegi_hazirla(ornek):
    """4 ending için (token matrisi, ending maskesi) döndürür; doğru etiket de verir."""
    ctx = enc.encode(ornek["ctx"])
    satirlar, maskeler = [], []
    for ending in ornek["endings"]:
        end = enc.encode(" " + ending)               # başına boşluk: GPT-2 kelimeleri böyle görür
        satirlar.append(ctx + end)
        maskeler.append([0] * len(ctx) + [1] * len(end))
    L = max(len(s) for s in satirlar)
    tok = torch.zeros((4, L), dtype=torch.long)
    mask = torch.zeros((4, L), dtype=torch.long)
    for i, (s, m) in enumerate(zip(satirlar, maskeler)):
        tok[i, :len(s)] = torch.tensor(s)
        mask[i, :len(m)] = torch.tensor(m)
    return tok, mask, int(ornek["label"])

@torch.no_grad()
def degerlendir(model, ornekler, ad):
    model.eval()
    dogru = dogru_norm = toplam = 0
    for ornek in ornekler:
        tok, mask, label = ornegi_hazirla(ornek)
        tok, mask = tok.to(device), mask.to(device)
        logits, _ = model(tok)
        logits = logits[..., :GERCEK_VOCAB]            # 50304'teki kullanılmayan token'ları dışarıda bırak
        # t. konumdaki logit, t+1. token'ı tahmin eder
        sl = logits[:, :-1, :].float()
        st = tok[:, 1:]
        sm = mask[:, 1:].float()
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1), reduction="none").view(4, -1)
        loss = loss * sm                                # sadece ending token'ları sayılır
        toplam_loss = loss.sum(dim=1)
        ort_loss = toplam_loss / sm.sum(dim=1)
        dogru += int(toplam_loss.argmin().item() == label)
        dogru_norm += int(ort_loss.argmin().item() == label)
        toplam += 1
        if toplam % 100 == 0:
            print(f"  [{ad}] {toplam} örnek | acc {dogru / toplam:.4f} | acc_norm {dogru_norm / toplam:.4f}",
                  flush=True)
    return dogru / toplam, dogru_norm / toplam, toplam

ornekler = veriyi_hazirla(args.yol)[:args.n]
print(f"değerlendirilen örnek sayısı: {len(ornekler)} | rastgele tahmin: %25\n")

sonuclar = {}

# 1) kendi modelim
kendi = GPT(GPTConfig(vocab_size=args.vocab))
kendi.load_state_dict(torch.load(args.model, map_location="cpu"))
kendi.to(device)
sonuclar["kendi model"] = degerlendir(kendi, ornekler, "kendi")
del kendi

# 2) OpenAI GPT-2 (124M)
openai = GPT.from_pretrained("gpt2").to(device)
sonuclar["OpenAI GPT-2"] = degerlendir(openai, ornekler, "openai")

print("\n" + "=" * 52)
print(f"{'model':<16}{'acc':>10}{'acc_norm':>12}{'örnek':>8}")
for ad, (a, an, n) in sonuclar.items():
    print(f"{ad:<16}{a:>10.4f}{an:>12.4f}{n:>8}")
print(f"{'rastgele':<16}{0.25:>10.4f}{0.25:>12.4f}")
print("=" * 52)

"""
- Senin modelin ne kadar buluyor, OpenAI'ın modelinden ne kadar uzakta? -

HellaSwag’i ilk 1000 örnekte, hem kendi modelimde hem OpenAI’ın GPT-2
ağırlıklarında çalıştırdım. Her örnekte bağlamın ardına 4 olası devamı
tek tek ekledim ve devam token’larının ortalama loss’u en düşük olanı modelin cevabı saydım.

Benim model acc_norm’da %23,6 buldu. Bu rastgele tahminin (%25) altında
ama 1000 örnekte standart hata yaklaşık %1,4 olduğu için aradaki fark
gürültü içinde. Yani modelim rastgeleden ayırt edilemiyor. acc %28,6 çıktı
ama acc toplam loss’a baktığı için kısa devamları kayırıyor, bu yüzden
acc_norm’a güveniyorum. OpenAI’ın GPT-2'si %33,6 buldu, yani rastgeleden
yaklaşık 9 puan yukarıda. Benimle arssında acc_norm’da yaklaşık 10 puan fark var.

Mimari aynı, ikisi de 124M parametre, yani fark boyuttan gelmiyor. Eğitimden geliyor.
Benim model 1 saatte yaklaşık 4,4 milyon token gördü ve val loss’u 6,5'te kaldı,
yani dilin temel yapısını yeni oturtuyor. OpenAI’ın modeli çok daha fazla veriyle,
çok daha uzun eğitildi. HellaSwag’de doğruyu seçmek için günlük olayların mantığını
bilmek gerekiyor. Bu da ancak çok fazla veriden öğreniliyor. Yani modelim “kötü” değil,
eğitimin çok erken bir noktasında. Daha çok token ve daha uzun eğitimle yukarı
çıkacağını düşünüyorum ama bunu denemedim.
"""