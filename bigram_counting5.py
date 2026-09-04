import torch
import matplotlib.pyplot as plt

def read_words(path="names.txt"):
    """
    karpathy'nin names.txt dosyasini okuyup her satiri bir isim olarak bir listeye
    topluyoruz. .strip() ile satir sonundaki bosluk/yeni-satir karakterlerini
    temizliyoruz.
    """
    with open(path, "r") as f:
        words = f.read().splitlines()
    return words

def build_vocab(words):
    """
    stoi (string to integer) ve itos (integer to string) sozluklerini
    kuruyor. Tum isimlerdeki benzersiz karakterleri bulup alfabetik
    siraya diziyoruz, 1'den baslayarak numaralandiriyoruz.
    '.' karakteri hem kelime baslangici hem bitisi anlamina geliyor,
    ona ozel olarak 0 indeksini ayiriyoruz.
    """
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def count_bigrams_dict(words):
    """
    ADIM 1: Bigram'lari once bir Python dictionary ile sayiyoruz.
    Her isme basina ve sonuna '.' ekleyip, zip(chs, chs[1:]) numarasiyla
    ardisik karakter ciftlerini (bigram) geziyoruz.
    """
    b = {}
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    return b

def count_bigrams_tensor(words, stoi, vocab_size):
    """
    ADIM 2: Ayni sayimi, ileride matematiksel islemler (normalize etme,
    olasiliga cevirme, orneklem cikarma) yapabilmek icin 27x27'lik bir
    torch tensor'unda tutuyoruz. Satirlar "su anki karakter",
    sutunlar "bir sonraki karakter".
    """
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    return N

def build_probability_table(N, smoothing=0):
    """
    ADIM 3: Sayim tablosunu (N) satir satir olasilik dagilimina ceviriyoruz.
    Her satiri KENDI TOPLAMINA boluyoruz. Boylece her satirin toplami 1
    oluyor (bir olasilik dagilimi).

    smoothing parametresi: her hucreye eklenen "sahte sayim". Bu sayede
    veri setinde hic gorulmemis bir bigram bile sifir olasilik almiyor.
    log(0) = -inf durumunu onluyor. smoothing=0 ise hic smoothing yok demek.

    KRITIK NOKTA: keepdim=True kullanmazsak, N.sum(1) sekli (27,) olur
    ve PyTorch'un broadcasting kurali bunu (1, 27) gibi yorumlar - yani
    SATIRLAR degil SUTUNLAR normalize edilir. Bu hicbir hata vermeden
    sessizce yanlis bir model uretir. keepdim=True ile sekil (27, 1)
    kalir ve broadcasting dogru sekilde "her satir kendi toplamina"
    bolunur.
    """
    P = (N + smoothing).float()
    P = P / P.sum(1, keepdim=True)
    return P

def verify_probability_rows(P, vocab_size):
    """
    Guvenlik agi: her satirin gercekten 1.0'a topladigini kontrol ediyoruz.
    keepdim hatasi yapilsaydi, bu kontrol tutmazdi.
    """
    row_sums = P.sum(1)
    hepsi_bir_mi = torch.allclose(row_sums, torch.ones(vocab_size))
    print(f"Her satir 1.0'a topluyor mu? {hepsi_bir_mi}")
    if not hepsi_bir_mi:
        print("UYARI: keepdim hatasi olabilir! Satir toplamlari:", row_sums)
    return hepsi_bir_mi

def negative_log_likelihood(words, P, stoi):
    """
    ADIM 5: Modelin kalitesini TEK BIR SAYIYLA olcuyoruz.

    1. Veri setindeki her bigram icin, modelin o bigram'a verdigi
       olasiligin LOGUNU aliyoruz (carpim yerine toplam kullanmak icin:
       log(a*b) = log(a) + log(b), sayisal olarak cok daha stabil).
    2. Tum bu log olasiliklari TOPLUYORUZ (log likelihood).
    3. NEGATIF'ini aliyoruz cunku olasiliklar hep <=1 oldugu icin
       log(p) hep negatif cikar, negatifini alinca "dusuk = iyi model"
       seklinde sezgisel bir loss elde ediyoruz (mse_loss'taki gibi).
    4. Bigram SAYISINA BOLEREK ORTALAMASINI aliyoruz. Boylece veri seti
       buyuklugunden bagimsiz, karsilastirilabilir bir sayi elde ediyoruz.
    """
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1

    nll = -log_likelihood
    average_nll = nll / n
    return average_nll.item(), n

def sample_name(P, itos, generator):
    """
    ADIM 4: Modelden tek bir isim orneklemek. '.' (indeks 0) ile basliyoruz,
    o satirin dagilimindan bir sonraki harfi orneklemek icin
    torch.multinomial kullaniyoruz. Orneklenen harf tekrar '.' (indeks 0)
    cikana kadar devam ediyoruz.
    """
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break
    return ''.join(out)

def visualize_bigram_table(N, itos, vocab_size, save_path="bigram_table5.png"):
    """
    27x27 tabloyu bir isi haritasi (heatmap) olarak ciziyor. Her hucreye
    hem hangi bigram oldugunu (ornegin 'ab') hem de kac kez gorundugunu
    yaziyoruz. Koyu mavi hucreler, cok sik gorulen bigram'lari gosteriyor.
    """
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    for i in range(vocab_size):
        for j in range(vocab_size):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=8)
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Bigram tablosu '{save_path}' olarak kaydedildi.")

def run_bigram_counting_model(words, dataset_name, save_prefix):
    """
    Tum sayma tabanli bigram pipeline'ini (vocab kurma, sayma, olasiliga
    cevirme, orneklem cikarma, NLL) tek bir veri seti icin calistiran
    yardimci fonksiyon. Boylece hem Ingilizce hem Turkce veriyi ayni
    kodla, tekrar yazmadan calistirabiliyoruz.
    """
    print("#" * 60)
    print(f"# {dataset_name}")
    print("#" * 60)
    print(f"Toplam isim sayisi: {len(words)}")
    print(f"Ilk 5 isim: {words[:5]}")
    print()

    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    print(f"Alfabe buyuklugu (nokta dahil): {vocab_size}")
    print("stoi:", stoi)
    print()

    # --- ADIM 1: Dictionary ile sayma ---
    b = count_bigrams_dict(words)
    en_sik_bigramlar = sorted(b.items(), key=lambda kv: -kv[1])
    print("En sik gorulen 10 bigram (dictionary ile):")
    for bigram, count in en_sik_bigramlar[:10]:
        print(f"  {bigram}: {count}")
    print()

    # --- ADIM 2: NxN tensor ile sayma ---
    N = count_bigrams_tensor(words, stoi, vocab_size)
    print("Tensor sekli:", N.shape)
    print()

    # --- Gorsellestirme ---
    visualize_bigram_table(N, itos, vocab_size, save_path=f"{save_prefix}_table.png")
    print()

    # --- ADIM 3: Sayim tablosunu olasiliga cevirme (smoothing'li) ---
    P = build_probability_table(N, smoothing=1)
    verify_probability_rows(P, vocab_size)
    print()

    # --- ADIM 4: Modelden isim orneklemek ---
    print("Modelden uretilen ornek isimler:")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(15):
        print(" ", sample_name(P, itos, g))
    print()

    # --- ADIM 5: Negative Log Likelihood ---
    nll_smoothed, n_bigrams = negative_log_likelihood(words, P, stoi)
    print(f"Ortalama NLL (smoothing +1): {nll_smoothed:.4f}  (toplam bigram: {n_bigrams})")
    print()

    return stoi, itos, N, P

if __name__ == "__main__":
    # --- Ingilizce veri seti (names.txt) ---
    words_en = read_words("names.txt")
    run_bigram_counting_model(words_en, "INGILIZCE ISIM LISTESI (names.txt)", "bigram_en")

    # --- Turkce veri seti (turkce_isimler.txt) ---
    words_tr = read_words("turkce_isimler.txt")
    run_bigram_counting_model(words_tr, "TURKCE ISIM LISTESI (turkce_isimler.txt)", "bigram_tr")