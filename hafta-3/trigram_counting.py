import random
import torch
import matplotlib.pyplot as plt
from bigram_counting import read_words, build_vocab

def split_dataset(words, train_frac=0.8, dev_frac=0.1, seed=42):
    """
    Veriyi train/dev/test olarak %80/%10/%10 boluyoruz.
    Once kelimeleri KARISTIRIYORUZ (shuffle) cunku names.txt
    alfabetik sirali olabilir. Karistirmazsak bolumler dengesiz olur.
    """
    words = words[:]  # orijinal listeyi bozmamak icin kopya
    random.Random(seed).shuffle(words)

    n = len(words)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    train_words = words[:n_train]
    dev_words = words[n_train:n_train + n_dev]
    test_words = words[n_train + n_dev:]

    return train_words, dev_words, test_words

def count_trigrams_tensor(words, stoi, vocab_size):
    """
    Trigram sayim tablosu. Bagalam (iki onceki karakter) bir CIFT
    oldugu icin, bu cifti tek bir indekse kodluyoruz:
        context_ix = ix1 * vocab_size + ix2
    Boylece tablo yine 2 boyutlu kaliyor: (vocab_size*vocab_size, vocab_size).
    Her kelimenin basina IKI nokta, sonuna BIR nokta ekliyoruz cunku
    ilk harfin bile 2 karakterlik bir baglami olmasi gerekiyor.
    """
    N = torch.zeros((vocab_size * vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            context_ix = ix1 * vocab_size + ix2
            N[context_ix, ix3] += 1
    return N

def build_probability_table(N, smoothing):
    """
    Bigram'daki ayni mantik: satir satir (bu sefer "satir" = bir
    baglam cifti) normalize ediyoruz. keepdim=True yine kritik.
    """
    P = (N + smoothing).float()
    P = P / P.sum(1, keepdim=True)
    return P

def negative_log_likelihood_trigram(words, P, stoi, vocab_size):
    """
    Bigram'daki NLL ile ayni mantik, sadece bagalam artik bir cift.
    """
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            context_ix = ix1 * vocab_size + ix2
            prob = P[context_ix, ix3]
            log_likelihood += torch.log(prob)
            n += 1
    nll = -log_likelihood
    return (nll / n).item(), n

def tune_smoothing(train_words, dev_words, stoi, vocab_size, candidates):
    """
    ADIM: smoothing gucunu DEV set'e gore ayarliyoruz. Train'den sayilan
    N tablosunu, her aday smoothing degeriyle olasiliga cevirip, DEV
    set'teki NLL'i olcuyoruz. En dusuk dev NLL'ini veren smoothing
    kazaniyor. Test set'e henuz dokunmuyoruz.
    """
    N_train = count_trigrams_tensor(train_words, stoi, vocab_size)

    best_smoothing = None
    best_dev_loss = float('inf')

    print("Smoothing taramasi (dev set uzerinde):")
    for s in candidates:
        P = build_probability_table(N_train, smoothing=s)
        dev_loss, _ = negative_log_likelihood_trigram(dev_words, P, stoi, vocab_size)
        print(f"  smoothing={s:6.2f}  ->  dev NLL = {dev_loss:.4f}")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_smoothing = s

    print(f"En iyi smoothing: {best_smoothing} (dev NLL = {best_dev_loss:.4f})")
    return best_smoothing, N_train

def sample_trigram(P, itos, vocab_size, generator):
    """
    Trigram modelinden isim orneklemek. Baglam iki karakter oldugu
    icin, '..' (iki nokta) ile basliyoruz, uretilen her yeni karakter
    bir onceki baglamin ikinci yarisi oluyor (kayan pencere/sliding window).
    """
    out = []
    ix1, ix2 = 0, 0  # baslangic baglami: iki nokta
    while True:
        context_ix = ix1 * vocab_size + ix2
        p = P[context_ix]
        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
        out.append(itos[ix3])
        if ix3 == 0:
            break
        ix1, ix2 = ix2, ix3  # pencereyi bir kaydir
    return ''.join(out)

if __name__ == "__main__":
    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    print("=" * 60)
    print("Veriyi train/dev/test olarak bolme (%80/%10/%10)")
    print("=" * 60)
    train_words, dev_words, test_words = split_dataset(words)
    print(f"Train: {len(train_words)}  Dev: {len(dev_words)}  Test: {len(test_words)}")
    print()

    print("=" * 60)
    print("Smoothing gucunu dev set'e gore ayarlama")
    print("=" * 60)
    candidates = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20]
    best_smoothing, N_train = tune_smoothing(train_words, dev_words, stoi, vocab_size, candidates)
    print()

    print("=" * 60)
    print("Nihai degerlendirme: TEST set uzerinde (sadece bir kez)")
    print("=" * 60)
    P_final = build_probability_table(N_train, smoothing=best_smoothing)
    test_loss_trigram, n_test = negative_log_likelihood_trigram(test_words, P_final, stoi, vocab_size)
    print(f"Trigram modeli test NLL: {test_loss_trigram:.4f}  (toplam trigram: {n_test})")
    print()

    # --- Karsilastirma: bigram modelinin ayni test seti uzerindeki loss'u ---
    print("=" * 60)
    print("Karsilastirma: bigram vs trigram (ayni test seti uzerinde)")
    print("=" * 60)
    from bigram_counting import count_bigrams_tensor, negative_log_likelihood

    N_bigram_train = count_bigrams_tensor(train_words, stoi, vocab_size)
    P_bigram = build_probability_table(N_bigram_train, smoothing=1)
    test_loss_bigram, _ = negative_log_likelihood(test_words, P_bigram, stoi)

    print(f"Bigram  modeli test NLL: {test_loss_bigram:.4f}")
    print(f"Trigram modeli test NLL: {test_loss_trigram:.4f}")
    if test_loss_trigram < test_loss_bigram:
        print("-> Trigram modeli daha dusuk loss veriyor: daha genis baglam")
        print("   (iki onceki harf) modele fayda sagliyor.")
    else:
        print("-> Bigram modeli bu veri setinde daha iyi/esit performans gosterdi;")
        print("   trigram'in fazladan parametresi, kucuk veri setinde asiri")
        print("   ogrenmeye (overfitting) yol acmis olabilir.")
    print()

    # --- Uretilen isimleri karsilastir ---
    print("=" * 60)
    print("Trigram modelinden uretilen ornek isimler")
    print("=" * 60)
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(15):
        print(" ", sample_trigram(P_final, itos, vocab_size, g))