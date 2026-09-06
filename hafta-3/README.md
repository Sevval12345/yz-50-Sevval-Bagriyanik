bigram_counting1.py  → Görev 1: sayma (dict + tensor) + görselleştirme
bigram_counting2.py  → Görev 2: olasılığa çevirme, keepdim, örnekleme
                        (bigram_counting1'den import)
bigram_counting3.py  → Görev 3: smoothing + NLL
                        (bigram_counting2'den import)
bigram_counting4.py  → Ara adım: sabit 27 → vocab_size (Türkçe'ye hazırlık)
                        (bigram_counting3'ten import)
bigram_counting5.py  → Görev 5: Türkçe veri + run_bigram_counting_model
                        (bigram_counting4'ten import)

bigram_counting.py   → Final/bağımsız konsolide versiyon
                        (diğer dosyalar bundan import ediyor — DEĞİŞMEDİ)
