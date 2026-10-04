import torch
import torch.nn.functional as F

torch.manual_seed(1337)

# Kucuk, elle takip edilebilir bir ornek: B=batch, T=zaman (baglam), C=kanal
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# Yöntem 1: for donguleriyle (acik, yavas ama tartismasiz dogru)
# Her (b, t) icin, 0'dan t'ye kadar olan tum gecmisin ortalamasini al.
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]       # (t+1, C): su ana kadarki tum gecmis
        xbow1[b, t] = xprev.mean(0)

# Yöntem 2: alt ucgen matrisle carpim
# wei'nin satiri [1/(t+1), ..., 1/(t+1), 0, 0, ...] seklinde: ilk t+1 pozisyona esit agirlik, geri kalanina sifir. wei @ x bu yuzden otomatik olarak kumulatif ortalamayi verir
wei2 = torch.tril(torch.ones(T, T))
wei2 = wei2 / wei2.sum(dim=1, keepdim=True)
xbow2 = wei2 @ x    # (T,T) @ (B,T,C) -> broadcast -> (B,T,C)


# Yöntem 3: softmax ile (gercek attention'in iskeleti)
# Once T x T'lik bir "ilgi puani" matrisi sifirlardan kuruyoruz (henuz ogrenilmis bir sey yok.
# Hepsi "esit ilgi" varsayimi). tril'in 0   "oldugu yerleri (gelecek pozisyonlar) -inf ile dolduruyoruz,
# boylece softmax sonrasi oralar tam olarak 0 olacak.
# Kalan (gecmis) pozisyonlar hepsi ayni degerde (0) oldugu icin softmax onlara esit olasilik verir.
# Sonuc yontem 2 ile birebir ayni.
# Farki: ileride bu sifirlarin yerine query@key.T'den gelen gercek, ogrenilmis puanlar gececek.
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
wei3 = wei3.masked_fill(tril == 0, float('-inf'))
wei3 = F.softmax(wei3, dim=-1)
xbow3 = wei3 @ x

# Dogrulama 
print("wei2 (ilk 4 satir, normalize edilmis tril):")
print(wei2[:4])
print("\nwei3 (ilk 4 satir, softmax sonrasi):")
print(wei3[:4])

print(f"\nxbow1 == xbow2 ? {torch.allclose(xbow1, xbow2, atol=1e-6)}")
print(f"xbow1 == xbow3 ? {torch.allclose(xbow1, xbow3, atol=1e-6)}")
print(f"xbow2 == xbow3 ? {torch.allclose(xbow2, xbow3, atol=1e-6)}")

""""
Matris carpimi aslinda satir bazinda bir dogrusal birlesim hesapliyor:
ciktinin her satiri, girdinin satirlarinin, agirlik matrisindeki
karsilik gelen satirla carpilip toplanmis hali. Eger o agirlik satirindaki
sayilar negatif degilse ve toplami 1 ise, bu dogrusal birlesim tam olarak
bir agirlikli ortalama tanimina uyuyor. torch.tril ile kurdugumuz
matriste her satirda sadece o ana kadarki pozisyonlara esit agirlik
veriliyor (digerleri 0), bu yuzden wei @ x satir satir "o ana kadarki
ortalama" uretiyor. Dongu yazmaya hic gerek kalmadan.
"""