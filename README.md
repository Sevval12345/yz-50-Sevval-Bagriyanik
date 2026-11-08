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
---

## Hafta 7: Self-Attention - GPT'nin Kalbi (Let's build GPT, Bölüm 1)

### Kaynaklar
- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Videodaki Colab Notebook'u](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [Andrej Karpathy - ng-video-lecture Reposu](https://github.com/karpathy/ng-video-lecture)
- [Tiny Shakespeare Verisi](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

### Haftanın Görevleri
1. **Veri Hazırlığı ve Karşılaştırma Tabanı:** Tiny Shakespeare'in okunması, karakter tokenizer'ının (`encode`, `decode`) yazılması, verinin %90 train / %10 val olarak ayrılması ve rastgele parçalardan batch çeken `get_batch` fonksiyonunun yazılması; Hafta 2'deki bigram modelinin `nn.Module` olarak kurulup eğitilmesi ve val loss'un karşılaştırma tabanı olarak kaydedilmesi. *(7:52 - 38:00)*
2. **Geçmişin Ortalaması (Üç Yol):** Geçmişin ortalamasının `for` döngüsüyle, lower triangular matrisle (`torch.tril`) çarparak ve softmax ile alınması; üç yolun aynı sonucu verdiğinin `torch.allclose` ile gösterilmesi ve matris çarpımının neden ağırlıklı ortalama olduğunun açıklanması. *(42:13 - 58:26)*
3. **Tek Bir Self-Attention Head:** Query, key, value ve pozisyon embedding'iyle tek bir head'in yazılması; bir örnekte ağırlık matrisinin (`wei`) yazdırılıp bir satırının okunması (token hangi token'lara ne kadar bakıyor); skorların `head_size`'ın kareköküne bölünmesinin etkisinin, bölmeden ve bölerek softmax çıktısı yan yana konarak sayıyla gösterilmesi. *(58:26 - 1:19:11)*
4. **Head'in Bigram Modeline Eklenmesi:** Head'in bigram modelinin içine konup eğitilmesi; val loss'un 1. görevdeki bigram tabanıyla karşılaştırılması ve düşüşün nedeninin yorumlanması; metin üretirken girdinin neden son `block_size` harfe kırpılması gerektiğinin açıklanması. *(1:19:11 - 1:21:59)*
5. **Ek Görev (a) - Attention Isı Haritası:** Eğitilmiş modelde bir cümlenin attention ağırlıklarının ısı haritası olarak çizilip hangi harfin hangisine baktığının yorumlanması.
6. **Ek Görev (b) - Maskesiz Model:** `tril` maskesi kaldırılarak modelin yeniden eğitilmesi; val loss ve üretilen metindeki değişimin gelecekteki harfleri görmekle (veri sızıntısı) açıklanması.
---
## Hafta 8: GPT'yi Tamamlamak - Multi-Head Attention ve Transformer Bloğu (Let's build GPT, Bölüm 2)

### Kaynaklar
- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out (1:21:59'dan sonrası)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Videodaki Colab Notebook'u](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [Andrej Karpathy - ng-video-lecture Reposu](https://github.com/karpathy/ng-video-lecture)
- [Andrej Karpathy - nanoGPT Reposu](https://github.com/karpathy/nanoGPT)
- [Vikikaynak - Türkçe Telifsiz Eserler](https://tr.wikisource.org/)
- [Attention Is All You Need (Google, 2017) (isteğe bağlı)](https://arxiv.org/abs/1706.03762)

### Haftanın Görevleri
1. **Multi-Head Attention:** Tek head'in aynı boyutu `n_head` tane küçük head'e bölen multi-head attention'a çevrilip çıktıların birleştirilmesi; val loss'un geçen haftaki tek head'le karşılaştırılması. *(1:21:59 - 1:24:25)*
2. **Transformer Bloğunun Tamamlanması:** Feedforward, residual bağlantı ve LayerNorm'un parça parça eklenip her parçadan sonra val loss'un kaydedilmesi; bigram, tek head, multi-head, feedforward, residual ve LayerNorm adımlarını içeren tek bir karşılaştırma tablosu çıkarılması; LayerNorm'un Hafta 6'daki BatchNorm1d ile karşılaştırılması (hangi eksende normalize ediyor, neden `running_mean` tutmuyor). *(1:24:25 - 1:37:49)*
3. **Modeli Büyütme ve Dropout:** Dropout eklenip donanıma sığan bir konfigürasyonun (`n_embd`, `n_head`, `n_layer`, `block_size`) seçilmesi; parametre sayısı, val loss ve eğitim süresinin raporlanması, seçimlerin gerekçelendirilmesi ve modelin ürettiği metinden örnek verilmesi. *(1:37:49 - 1:42:39)*
4. **Türkçe Metinle Eğitim:** Aynı modelin, Tiny Shakespeare'e yakın boyutta (yaklaşık 1 MB) kendi derlenen Türkçe metinle eğitilmesi; üretilen metinden örnek verilmesi; Türkçe metnin farklı karakter sayısının Shakespeare'inkiyle karşılaştırılması ve iki val loss'un doğrudan karşılaştırılıp karşılaştırılamayacağının tartışılması.
----

## Hafta 9: Tokenizer - BPE'yi Sıfırdan Yazmak (Let's build the GPT Tokenizer)

### Kaynaklar
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Videodaki Colab Notebook'u](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [Andrej Karpathy - minbpe Reposu](https://github.com/karpathy/minbpe) ve [Alıştırma Dosyası](https://github.com/karpathy/minbpe/blob/master/exercise.md)
- [tiktokenizer - Metni Tarayıcıda Token'larına Ayıran Araç](https://tiktokenizer.vercel.app)
- [OpenAI - tiktoken Kütüphanesi](https://github.com/openai/tiktoken)
- [GPT-2 Makalesi, 2.2 "Input Representation" (isteğe bağlı)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### Haftanın Görevleri
1. **Metni Byte'lara Çevirme:** Türkçe harfler içeren (ç, ğ, ı, ö, ş, ü) bir cümle seçilip her harfin Unicode kod noktasının ve UTF-8 byte'larının yazdırılması; hangi harflerin 1, hangilerinin 2 byte tuttuğunun incelenmesi ve kod noktalarını doğrudan vocab olarak kullanmanın neden kötü bir fikir olduğunun açıklanması. *(0:14:56 - 0:23:50)*
2. **BPE'nin Sıfırdan Yazılması ve Türkçe Metinde Eğitilmesi:** Ardışık ikililerin sayılıp en sık geçenin yeni bir token'la birleştirilmesi, istenen vocab boyutuna (512) gelene kadar tekrarlanması; ilk 20 birleştirmenin metin olarak yazdırılıp hangi harf gruplarının önce birleştiğinin incelenmesi ve sıkıştırma oranının (byte sayısı / token sayısı) hesaplanması. *(0:23:50 - 0:39:20)*
3. **`encode` ve `decode`:** `decode(encode(metin))` işleminin Türkçe karakterli ve emojili bir örnekle metnin kendisini verdiğinin gösterilmesi; geçersiz UTF-8 token dizilerinde `decode`'un ne yapması gerektiğinin ve nedeninin açıklanması. *(0:42:47 - 0:57:36)*
4. **GPT-4 Regex Deseni:** Metnin önce GPT-4'ün deseniyle parçalara bölünüp BPE'nin her parçada ayrı çalıştırılması; tokenizer'ın yeniden eğitilip birleştirmelerin öncekiyle karşılaştırılması; "Ankara'dan" ve "İstanbul'da" gibi kesme işaretli kelimelerin bu desenle bölünüp İngilizce için yazılmış desenin Türkçede ne yaptığının yorumlanması. *(0:57:36 - 1:11:38)*
5. **Türkçe ve İngilizce Karşılaştırması:** Aynı paragrafın Türkçesinin ve İngilizcesinin GPT-4 tokenizer'ıyla (`tiktoken`, `cl100k_base`) token'lanıp kelime başına düşen token sayısının bulunması; aynı Türkçe paragrafın kendi tokenizer'la token'lanması ve Türkçenin eklemeli yapısının sayılara etkisinin yorumlanması.
---

## Hafta 12: GPT-2'yi Gerçek Veriyle Eğitmek ve Ölçmek

### Kaynaklar
- [Andrej Karpathy - Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Andrej Karpathy - build-nanogpt Reposu](https://github.com/karpathy/build-nanogpt)
- [FineWeb-Edu Veri Seti (Hugging Face)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HellaSwag: Can a Machine Really Finish Your Sentence? (arXiv)](https://arxiv.org/abs/1905.07830)
- [HellaSwag Veri Seti (Hugging Face, Rowan/hellaswag)](https://huggingface.co/datasets/Rowan/hellaswag)

### Haftanın Görevleri
1. **Veriyi Hazırla:** FineWeb-Edu'dan 25 milyon token'ın GPT-2 tokenizer'ıyla token'lanıp `uint16` shard dosyalarına bölünmesi (ilk shard doğrulama için ayrıldı) ve bu shard'ları sırayla okuyan bir data loader'ın yazılması. *(3:10:21 - 3:23:10)*
   - `prepare_data.py`: FineWeb-Edu'yu `streaming` ile akıtıp her dokümanın başına `<|endoftext|>` koyarak token'lıyor, 5 milyonluk shard'lara yazıyor. **Çıktı:** 5 shard, 25 milyon token, diskte 50 MB (token başına 2 byte, `uint16` ile uyumlu). 1 val (5M) + 4 train (20M) shard.
   - `data_loader.py`: `x` ve bir kaydırılmışı `y`'yi üretiyor, shard bitince bir sonrakine geçiyor. **Çıktı:** train ve val aynı dosyayı paylaşmıyor, `y`'nin `x`'in kaydırılmışı olduğu doğrulandı (`True`), token aralığı 0–50256, ilk shard 4,882 batch sürüp ikinci shard'a sorunsuz geçti.
2. **Doğrulama Loss'u ve Örnekleri Ekle:** Eğitim döngüsüne belirli aralıklarla val loss hesabı ve örnek metin üretimi eklenmesi; train ve val loss'larının aynı grafikte çizilmesi. *(3:23:10 - 3:28:23)*
   - `train_gpt2.py` (Hafta 10-11'deki model, ayarlar ve gradient accumulation tek scriptte): `val_loss_hesapla` val shard'ının başından sabit 20 batch üzerinden ortalama loss'u, `ornek_uret` sabit seed'li top-k örnekleme yapıyor. **Çıktı (100 adım, 228 sn):** val loss 11.01 → 7.87 → 7.63 → 7.51 → 7.45. Başlangıç değeri rastgele tahminin kayıp değerine (`ln 50304 ≈ 10.83`) yakın. Örnekler önce kelime çorbası, sonra sık kelimeler ve noktalama ("and, of, the"), sonra "the the the" tekrarı.
3. **Modeli Eğit:** Modelin sabit bir süre (1 saat) eğitilip görülen token, token/sn ve son loss'un raporlanması; eğitim ilerledikçe üretilen metnin nasıl değiştiğinin yazılması. *(3:28:23 - 3:43:05)*
   - `train_gpt2.py --sure 3600`: **Çıktı:** 1,065 adım, 4,362,240 token, 3,602.9 sn, ortalama 1,211 token/sn. Val loss 11.01 → 6.50, son train loss ortalaması 6.49. Train ≈ val: veri setinin 20M'lik train kısmının yalnızca ~4.4M token'ı görüldü, ezberleme yok.
   - **Metin nasıl değişti:** Önce rastgele token'lar, 100. adımda gerçek ama sık kelimeler ve rastgele noktalama, 200-300. adımda cümleye benzeyen kalıplar, 400-600. adımda konular (çocuk, su, sağlık), sonrasında tırnak işaretleri ve `<|endoftext|>`. Bir saatin sonunda bile metin anlamlı değil. İlk 300 adımdaki değişim çok daha belirgin, son 400 adımda loss'taki küçük düşüş metinde gözle zor ayırt ediliyor.
4. **HellaSwag'de Ölç:** HellaSwag'in (ilk 1000 örnek) önce kendi modelde, sonra OpenAI'ın GPT-2 ağırlıklarında çalıştırılması; rastgele tahminin %25 olduğu hatırlanarak aradaki farkın nedeninin yazılması. *(3:28:23 - 3:56:21)*
   - `hellaswag_eval.py`: her örnekte bağlama 4 devamı ayrı ayrı ekleyip sadece devam token'larının loss'una bakıyor; `acc` (loss toplamı) ve `acc_norm` (loss ortalaması). **Çıktı (ilk 1000 örnek):** kendi model `acc` 0.286 / `acc_norm` 0.236, OpenAI GPT-2 `acc` 0.335 / `acc_norm` 0.336, rastgele 0.25. 1000 örnekte standart hata ≈ ±%1.4.
   - **Yorum:** Kendi modelim `acc_norm`'da rastgeleden ayırt edilemiyor. OpenAI'ınki rastgeleden net yukarıda, fark ≈ 10 puan. Mimari aynı (124M), fark eğitimden: benim model ~4.4M token görüp val loss 6.5'te kaldı, OpenAI'ınki çok daha fazla veriyle eğitildi.
5. **Ek Görev - Türkçe Model:** Aynı modelin Türkçe metinde eğitilip örnek metin alınması; HellaSwag İngilizce olduğu için Türkçe modelin nasıl ölçüleceğinin yazılması.
   - `prepare_turkish.py`: Hafta 9'daki `turkce_metin2.txt`'yi GPT-2 tokenizer'ıyla token'layıp son %10'u val ayırıyor. **Çıktı:** 455,792 token (train 410,213 / val 45,579), token başına 2.12 byte, 1 epoch = 100 adım.
   - `train_gpt2.py --steps 600 --prompt "Merhaba, ben"`: **Çıktı:** 600 adım (≈ 6 epoch), 36.8 dk, val loss 10.94 → 3.57, son train loss ort. 2.80. Val yükselmedi ama train'den ~0.77 uzaklaştı ve son 100 adımda neredeyse düz kaldı: ezberlemenin başlangıcı gibi. Train loss her ~100 adımda (epoch sınırında) sıçrıyor, loader karıştırmadan okuduğu için metnin aynı zor bölgesine denk geliyor olabilir (doğrulanmadı). Üretilen metin anlamsız ama Türkçe'nin yapısı var (gerçek kelimeler, ekler, noktalama, diyalog biçimi).
   - `turkish_measure.py`: tutulmuş metinde loss ve bits-per-byte, ayrıca HellaSwag mantığında kendi mini testi. **Çıktı:** val loss 3.559 (perplexity 35.1), **2.424 bits-per-byte**, mini test 178/500 = **%35.6** (rastgele %25, ±%1.9). Test kolay olduğu için %36 düşük, model 64 token'lık bağlamı pek kullanmıyor. İngilizce modelin loss'uyla kıyaslanmaz (farklı veri, Türkçe val aynı metnin son %10'u).
   - **Türkçe nasıl ölçülür:** Byte başına bit (tokenizer'dan bağımsız kıyas) ve HellaSwag mantığında çoktan seçmeli test. Gerçek ölçüm için Türkçe'ye çevrilmiş HellaSwag ya da Türkçe içeren bir benchmark gerekir, bunların varlığını kontrol etmedim.
