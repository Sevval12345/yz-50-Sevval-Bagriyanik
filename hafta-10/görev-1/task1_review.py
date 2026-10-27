# Görev 1: Hazır GPT-2 (124M) ağırlıklarını incele.
# İlk çalıştırmada ~500 MB ağırlık indirir.

import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")   # "gpt2" = 124M sürümü
sd_hf = model_hf.state_dict()

# ------------------------------------------------------------------
print("=" * 78)
print("1) HER AĞIRLIĞIN ADI VE BOYUTU")
print("=" * 78)
for name, tensor in sd_hf.items():
    print(f"{name:<45}{str(tuple(tensor.shape)):<20}{tensor.numel():>12,}")

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("2) TOPLAM PARAMETRE SAYISI")
print("=" * 78)
n_params = sum(p.numel() for p in model_hf.parameters())
n_state = sum(t.numel() for t in sd_hf.values())
print(f"model.parameters() toplamı : {n_params:,}")
print(f"state_dict toplamı         : {n_state:,}")
print(f"fark                       : {n_state - n_params:,}")
print(f"wte boyutu (50257 x 768)   : {sd_hf['transformer.wte.weight'].numel():,}")
print(f"lm_head ile wte aynı tensör mü?: "
      f"{model_hf.lm_head.weight.data_ptr() == model_hf.transformer.wte.weight.data_ptr()}")

# ------------------------------------------------------------------
print("\n" + "=" * 78)
print("3) wpe (POZİSYON EMBEDDING) GRAFİĞİ")
print("=" * 78)
wpe = sd_hf["transformer.wpe.weight"]
print(f"wpe boyutu: {tuple(wpe.shape)}  (1024 pozisyon, her biri 768 boyutlu)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Sol: tabloyu olduğu gibi göster (satır = pozisyon, sütun = kanal)
im = axes[0].imshow(wpe.detach().numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
axes[0].set_xlabel("kanal (0..767)")
axes[0].set_ylabel("pozisyon (0..1023)")
axes[0].set_title("wpe: tüm tablo")
fig.colorbar(im, ax=axes[0])

# Sağ: 3 kanal pozisyon boyunca ne yapıyor?
for ch in (150, 200, 250):
    axes[1].plot(wpe[:, ch].detach().numpy(), label=f"kanal {ch}")
axes[1].set_xlabel("pozisyon")
axes[1].set_ylabel("değer")
axes[1].set_title("üç kanalın pozisyon boyunca değişimi")
axes[1].legend()

plt.tight_layout()
plt.savefig("wpe.png", dpi=150)
print("Grafik 'wpe.png' olarak kaydedildi.")
plt.close()

"""
Pozisyon embedding'lerini (wpe) çiz: ne görüyorsun, kendi cümlelerinle yaz.

- Tepeler ve çukurlar var, rastgele değil. Üç kanal da pozisyon boyunca
yavaşça yükselip alçalıyor. Eğer tablo rastgele kalsaydı düz gürültü görürdük.
- Temiz sinüs yok. Dalgalar çok düzgün değil, yavaş ve düzensiz. Üstüne ince gürültü biniyor.
- İlk birkaç pozisyon özel. Kanal 250 pozisyon 0'da yaklaşık -0.1'e düşüp
hızla toparlanıyor. Bu mantıklı, çünkü pozisyon 0 modelin “metnin başı” olduğunu anladığı yer.
- Sona doğru (1000+) üç kanal da yukarı çıkıyor. Büyük pozisyonlar eğitimde az görüldüğü
için bu bölgeler daha az oturmuş olabilir. Bunu kesin söyleyemem, tahmin olarak yaz.
Yorum: Pozisyon bilgisi elle tasarlanmadı, modelin kendisi öğrendi ve yakın pozisyonlar
benzer değerler aldığı için tablo pürüzsüz bir yapıya yakınsadı.
“Attention Is All You Need”deki sabit sinüs/kosinüs fikrine benzer bir şeyi kendi başına bulmuş.
"""