# YZ-50 Yapay Zeka Çalışmaları
**https://yz50.ai/**

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
---

## Hafta 4: MLP Dil Modeli & Eğitim İçgörüleri (Bengio 2003 - makemore Part 2 & 3)

### Kaynaklar
- [Andrej Karpathy - Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Andrej Karpathy - Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [Bengio et al. - A Neural Probabilistic Language Model (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Andrej Karpathy - makemore Reposu](https://github.com/karpathy/makemore)

### Haftanın Görevleri
1. **Veri Seti & Embedding:** Önceki üç harfi bağlam alan (X: 3 harf indeksi, Y: sıradaki harf) veri setinin kurulması; 27x2 boyutunda embedding tablosunun oluşturulup indeksleme ile embedding'lerin çekilmesi. *(Part 2, 9:03 - 18:35)*
2. **Gizli Katman & Çıkış Katmanı:** Embedding'lerin düzleştirilip W1/b1 ile tanh, W2/b2 ile logits üretilmesi; loss'un elle hesaplanıp `F.cross_entropy` ile karşılaştırılması ve sayısal stabilite avantajının gösterilmesi. *(18:35 - 37:56)*
3. **Eğitim Döngüsü:** Tek bir minibatch'in overfit edilmesi (sağlık kontrolü), learning rate taraması ile iyi bir değer seçilmesi, verinin train/dev/test olarak bölünüp dev loss'un raporlanması. *(37:56 - 1:00:49)*
4. **Kapasite Artırımı & Görselleştirme:** Gizli katman ve embedding boyutunun büyütülüp dev loss değişiminin incelenmesi; embedding'lerin 2 boyutta çizdirilip karakterler arası yakınlığın yorumlanması; modelden isim örneklenip bigram sonuçlarıyla karşılaştırılması. *(1:00:49 - 1:13:24)*
5. **Başlangıç Loss'u & Tanh Doyması:** Başlangıç loss'unun neden çok yüksek olduğunun ve tanh'ın neden doyduğunun histogramlarla gösterilmesi; Kaiming init ile ağırlıkların ölçeklenip iki sorunun da düzeltilmesi. *(Part 3, 4:19 - 40:40)*
6. **BatchNorm:** Gizli katmandan sonra BatchNorm katmanı eklenmesi; eğitimde batch istatistiği, tahminde running mean/std kullanılması; BatchNorm'lu ve BatchNorm'suz modelin dev loss'unun karşılaştırılması. *(40:40 - 1:04:50)*
7. **Türkçe Karakter Genişletmesi:** Aynı modelin geçen haftaki Türkçe isim listesiyle eğitilmesi; üretilen isimlerin ve dev loss'un bigram'ın Türkçe sonuçlarıyla karşılaştırılması.
8. **Ek Görev (Hiperparametre Optimizasyonu):** Embedding boyutu, gizli katman genişliği, learning rate decay gibi hiperparametrelerin ayarlanarak Karpathy'nin videodaki 2.2 validation loss referansının geçilmesi.
---

## Hafta 5: Backprop Ninja - Autograd'ı Elle Yeniden İnşa Etmek (makemore Part 4)

### Kaynaklar
- [Andrej Karpathy - Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Videodaki egzersiz notebook'u](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)
- [Andrej Karpathy - makemore Reposu](https://github.com/karpathy/makemore)

### Haftanın Görevleri
1. **Modeli Atomik Adımlara Bölme:** Hafta 4'teki MLP + BatchNorm modelinin `logits, counts, probs, logprobs` gibi ara değişkenlere bölünmesi; `retain_grad()` ile her ara değişkenin gradyanının `loss.backward()` üzerinden PyTorch'a hesaplatılması. *(Egzersiz 1, videonun ilk yarısı)*
2. **Elle Gradyan Hesaplama & Doğrulama:** Aynı ara değişkenlerin gradyanlarının zincir kuralıyla elle türetilmesi; `cmp` fonksiyonuyla PyTorch'un gradyanlarıyla tek tek karşılaştırılıp `exact`/`approximate` sonucuna ulaşılması, broadcasting'in tersinin (toplama/yayma) ve çoklu yoldan gelen gradyanların doğru şekilde ele alınması.
3. **Ek Görev (Tek İfadeye İndirgeme):** Cross entropy'nin ve BatchNorm'un geriye yayılımının cebirsel olarak tek bir ifadeye sadeleştirilmesi; bu kısayollarla, `loss.backward()` hiç çağrılmadan, tamamen elle hesaplanan gradyanlarla modelin eğitilmesi. *(Egzersiz 2, 3, 4)*
---

## Hafta 6: torch.nn'i Sıfırdan Yazmak & WaveNet (makemore Part 5)

### Kaynaklar
- [Andrej Karpathy - Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- [Videodaki notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
- [Andrej Karpathy - makemore reposu](https://github.com/karpathy/makemore)
- [WaveNet: A Generative Model for Raw Audio (DeepMind, 2016)](https://arxiv.org/abs/1609.03499)

### Haftanın Görevleri
1. **Kendi `torch.nn` Katmanları:** `Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential` sınıflarının sıfırdan yazılması; modelin tek bir `Sequential` olarak kurulup eğitim döngüsünün katman isimlerinden bağımsız hale getirilmesi; loss eğrisi çiziminin gruplanmış ortalamayla düzeltilmesi. *(1:40 - 17:11)*
2. **Bağlamı Büyütme (Karşılaştırma Tabanı):** Bağlam penceresinin 3 harften 8 harfe çıkarılması, mimari değiştirilmeden aynı düz modelin koşulup dev loss'un kaydedilmesi; parametre sayısındaki artışın ve loss'taki değişimin raporlanması. *(17:11 - 21:36)*
3. WaveNet'in Kurulması: 8 harfin tek seferde düzleştirilmesi yerine ikişer ikişer, üç katmanda hiyerarşik olarak tek vektöre indirilmesi (`FlattenConsecutive`); her katmanın çıktı şeklinin yazdırılıp nedeninin açıklanması. *(21:36 - 37:41)*
4. BatchNorm1d'nin 3 Boyutlu Hatası: `BatchNorm1d`'nin üç boyutlu girdide (batch, zaman, kanal) yalnızca batch ekseninde ortalama almasının yol açtığı hatanın tespit edilmesi; ortalamanın doğru eksenlerde (`batch, zaman`) alınacak şekilde düzeltilmesi; düzeltme öncesi ve sonrası dev loss'un karşılaştırılması. *(37:41 - 46:07)*
5. **Modeli Büyütme & Karşılaştırma Tablosu:** Embedding boyutu ve katman genişliği artırılarak üç satırlık bir tablo çıkarılması: bağlam 3 düz MLP, bağlam 8 düz MLP, bağlam 8 WaveNet - parametre sayısı ve dev loss sütunlarıyla. *(46:07 - 46:58)*
6. **Türkçe Karakter Genişletmesi:** Aynı WaveNet mimarisinin Türkçe isim listesiyle eğitilmesi; üretilen isimlerden örnekler verilip dev loss'un hafta 4'ün Türkçe MLP'siyle karşılaştırılması; bağlamı 8'e çıkarmanın Türkçe'de kazandırdığının değerlendirilmesi.
