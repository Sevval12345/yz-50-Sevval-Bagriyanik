## Dosya Yapısı ve Görev Dağılımı

| Dosya Adı | Açıklama | Bağımlılık / Not |
| :--- | :--- | :--- |
| **`bigram_counting1.py`** | **Görev 1:** Karakter çiftlerini sayma (`dict` + `tensor`) ve görselleştirme. | Başlangıç adımı |
| **`bigram_counting2.py`** | **Görev 2:** Frekansları olasılığa çevirme, `keepdim` kullanımı ve modelden örnekleme (sampling). | `bigram_counting1`'den import eder |
| **`bigram_counting3.py`** | **Görev 3:** Model düzleştirme (*Laplace smoothing*) ve Negatif Log-Likelihood (NLL) kaybı hesaplama. | `bigram_counting2`'den import eder |
| **`bigram_counting4.py`** | **Ara Adım:** Sabit boyutlu yapıdan (`27`) dinamik `vocab_size` yapısına geçiş (Türkçe hazırlığı). | `bigram_counting3`'ten import eder |
| **`bigram_counting5.py`** | **Görev 5:** Türkçe veri seti entegrasyonu ve `run_bigram_counting_model` fonksiyonu. | `bigram_counting4`'ten import eder |
| **`bigram_counting.py`** | **Final:** Tüm adımların bir arada bulunduğu bağımsız ve konsolide versiyon. | Diğer dosyalar tarafından referans alınır |
