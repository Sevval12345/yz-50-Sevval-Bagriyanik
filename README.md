# YZ-50 Yapay Zeka Çalışmaları

YZ-50 süreci boyunca tamamlanan haftalık görevler, kodlar ve referans kaynaklar yer almaktadır.

---
## Hafta 1: Yapay Sinir Ağlarına Giriş & Gradient Descent

### Kaynaklar
- [3Blue1Brown - But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [3Blue1Brown - Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Andrej Karpathy - The spelled-out intro to neural networks (İlk 19 dk - Sayısal Türev)](https://www.youtube.com/watch?v=VMj-3S1tku0)

### Haftanın Görevleri
1. **Tek Nöron Forward Pass:** Python ile harici kütüphane kullanmadan tek nöronluk forward pass implementasyonu.
2. **Katman Mimarisi:** Birden fazla nörondan oluşan küçük bir katman kurarak forward pass'in genişletilmesi.
3. **Loss Fonksiyonu:** Model performansını ölçen basit bir kayıp (loss) fonksiyonunun yazılması.
4. **Loss Eğrisi:** Parametreler manuel değiştirilerek loss değişiminin incelenmesi ve loss eğrisinin çizdirilmesi.
5. **Gradient Descent:** Sayısal türev (numerical derivative) kullanılarak parametrelerin güncellendiği temel bir optimizasyon döngüsünün kurulması.
---

## Hafta 2: Backpropagation & Micrograd Mimarisi

### Kaynaklar
- [Andrej Karpathy - The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Andrej Karpathy - micrograd Reposu](https://github.com/karpathy/micrograd)
- [3Blue1Brown - What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [3Blue1Brown - Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)

### Haftanın Görevleri
1. **`Value` Sınıfı Kurulumu:** Toplama ve çarpma işlemleriyle başlayıp türeyen değerleri ve operasyon geçmişini saklayan veri yapısının kurulması (Graphviz ile hesaplama grafiği görselleştirme). *(19:09 - 32:10)*
2. **Manuel Gradient Hesaplama:** Basit matematiksel ifadeler ve tek bir nöron (tanh eklenerek) üzerinde zincir kuralının (chain rule) elle hesaplanarak kavranması. *(32:10 - 1:09:02)*
3. **`backward()` Metodu:** Çıktı gradyanını 1 kabul edip ters topolojik sıralama ile tüm düğümlere zincir kuralının işletilmesi ve çoklu dallanmalarda gradyanların toplanarak biriktirilmesi. *(1:09:02 - 1:27:05)*
4. **Operasyon Parçalama ve Doğrulama:** `tanh` fonksiyonunun `exp`, bölme ve üs alma operasyonlarına ayrılarak türevinin doğrulanması; sonuçların `backward()`, sayısal türev ve PyTorch çıktılarıyla karşılaştırılması. *(1:27:05 - 1:43:55)*
5. **Neuron, Layer ve MLP Katmanları:** Modüler sinir ağı sınıflarının inşası, parametrelerin toplanması, küçük bir veri kümesinde loss'un düşürülmesi ve gradyan sıfırlama (zero_grad) adımının uygulanması. *(1:43:55 - 2:14:03)*
---

## Hafta 3: Dil Modelleme (Bigram & Trigram - makemore)

### Kaynaklar
- [Andrej Karpathy - The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Andrej Karpathy - makemore Reposu](https://github.com/karpathy/makemore)
- [PyTorch Broadcasting Semantics Dokümantasyonu](https://pytorch.org/docs/stable/notes/broadcasting.html)

### Haftanın Görevleri
1. **Bigram Sayım Tablosu:** İngilizce `names.txt` verisiyle bigram çiftlerinin önce Python sözlüğü, ardından 27x27 boyutunda PyTorch tensörü üzerinde sayılması ve görselleştirilmesi. *(03:03 - 24:02)*
2. **Örnekleme & Olasılık Dağılımı:** Sayım tablosunun satır bazında normalleştirilerek olasılıklara dönüştürülmesi ve modelden yeni isimler üretilmesi (`keepdim` ve broadcasting kurallarına dikkat edilerek). *(24:02 - 50:14)*
3. **Negative Log Likelihood (NLL) & Smoothing:** Model başarımının NLL ile değerlendirilmesi ve sıfır olasılık problemini önlemek için Laplace smoothing (sahte sayım) eklenmesi. *(50:14 - 1:02:57)*
4. **Sinir Ağı Tabanlı Bigram:** One-hot encoding girdi, 27x27 ağırlık matrisi, Softmax aktivasyonu, NLL loss ve gradient descent döngüsü ile tek katmanlı yapay sinir ağının kurulup sayım modeli loss'una yakınsamasının incelenmesi. *(1:02:57 - 1:54:31)*
5. **Türkçe Karakter Genişletmesi:** Alfabenin Türkçe karakterlerle (`ç, ğ, ı, ö, ş, ü`) genişletilerek açık kaynak Türkçe isim veri kümesi üzerinde her iki yaklaşımın (sayım & sinir ağı) eğitilmesi ve üretilen isimlerin incelenmesi.
6. **Ek Görev (Trigram Modeli):** İki önceki harfi dikkate alan trigram modelinin geliştirilmesi; verinin Train / Dev / Test (%80 / %10 / %10) olarak ayrılıp geliştirme kümesi kaybına göre smoothing hiperparametre optimizasyonu yapılması.
