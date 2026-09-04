import torch
import torch.nn.functional as F
from bigram_counting import read_words, build_vocab

def build_training_set(words, stoi):
    """
    Egitim verisini kuruyoruz: her bigram icin xs (ilk karakterin indeksi)
    ve ys (ikinci/dogru karakterin indeksi) listelerini olusturuyoruz.
    Bu, önceki gorevdeki NLL hesaplamasinda gezdigimiz ayni bigram'lar
    sadece artik bir for dongusu yerine tensor olarak tutuluyor.
    """
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys

def init_weights(vocab_size, seed=2147483647):
    """
    27x27'lik agirlik matrisini rastgele baslatiyoruz. requires_grad=True
    ile PyTorch'a bu tensorun gradyanini takip etmesini soyluyoruz.
    Gecen hafta (2. hafta) Value sinifinda grad alaninin otomatik doldurulmasiyla
    ayni amac.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    return W

def forward_pass(xs, W, vocab_size):
    """
    ADIM 1: One-hot encoding + softmax ile forward pass.

    1. xenc: her karakteri 27 boyutlu one-hot vektore ceviriyoruz
       (sadece o karakterin pozisyonunda 1, geri kalani 0).
    2. logits = xenc @ W: matris carpimi, "log-counts" gibi dusunulebilir.
       one-hot vektor sayesinde bu carpim aslinda W'nin ilgili satirini
       "seciyor".
    3. counts = logits.exp(): negatif olabilecek sayilari pozitife
       ceviriyor. N tablosundaki ham sayimlara benzer bir yapi.
    4. probs = counts / counts.sum(1, keepdim=True): satir satir
       normalize ederek olasilik dagilimina ceviriyoruz.
       (keepdim=True kritik. Onceki gorevdeki ayni broadcasting kurali)

    Bu uc adim (exp + normalize) birlikte SOFTMAX fonksiyonunu olusturuyor.
    """
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    return probs

def compute_loss(probs, ys, W, reg_strength=0.01):
    """
    ADIM 2: Loss hesaplama. Onceki gorevdeki NLL ile matematiksel
    olarak BIREBIR AYNI, sadece artik vektorize (tensor indeksleme ile,
    for dongusu olmadan) yapiyoruz.

    probs[torch.arange(num), ys]: her ornek icin modelin DOGRU cevaba
    (ys) verdigi olasiligi seciyor. torch.arange(num) satir indeksleri,
    ys ise her satirda hangi sutuna (dogru sonraki karaktere) bakilacagini
    soyluyor.

    reg_strength * (W**2).mean(): kucuk bir regularization terimi.
    W'nin cok buyuk degerlere gitmesini cezalandiriyor, sayma
    modelindeki SMOOTHING'in sinir agi karsiligi gibi dusunulebilir.
    W sifira yakin kaldikca, olasilik dagilimi daha uniform (duzgun) oluyor.
    """
    num = probs.shape[0]
    nll = -probs[torch.arange(num), ys].log().mean()
    reg = reg_strength * (W ** 2).mean()
    return nll + reg

def train(xs, ys, W, vocab_size, steps=200, learning_rate=50.0, reg_strength=0.01, verbose_every=20):
    """
    ADIM 3: Gradient descent dongusu. Gecen hafta (2. hafta) micrograd'da yazdigimiz
    backward() mekanizmasinin, PyTorch tarafindan buyuk tensorler
    uzerinde otomatik calistirilmis hali. Kavramsal olarak birebir aynilar:
    forward pass -> loss hesapla -> backward pass -> parametreleri guncelle.
    """
    for k in range(steps):
        # --- forward pass ---
        probs = forward_pass(xs, W, vocab_size)
        loss = compute_loss(probs, ys, W, reg_strength)

        # --- gradyanlari sifirla ---
        # (Value.parameters()'daki p.grad = 0.0 ile ayni amac)
        W.grad = None

        # --- backward pass ---
        # loss.backward(), gecen hafta kendi yazdigin Value.backward()
        # ile AYNI mekanizma: topolojik siralama + zincir kurali,
        # sadece burada PyTorch bunu (27,27) boyutundaki W tensoru
        # icin otomatik yapiyor.
        loss.backward()

        # --- update: parametreyi gradyanin TERSI yonunde kaydir ---
        W.data += -learning_rate * W.grad

        if k % verbose_every == 0 or k == steps - 1:
            print(f"adim {k:3d}  loss = {loss.item():.4f}")

    return W

def sample_from_nn(W, itos, vocab_size, generator):
    """
    Egitilmis sinir agi modelinden bir isim orneklemek. Sayma
    modelindeki sample_name ile ayni mantik, sadece olasiliklari
    W'den (forward_pass ile) hesapliyoruz.
    """
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break
    return ''.join(out)

def run_bigram_nn_model(words, dataset_name, counting_model_loss):
    """
    Tum sinir-agi-tabanli bigram pipeline'ini (egitim verisi kurma,
    agirlik baslatma, gradient descent, ornekleme) tek bir veri seti
    icin calistiran yardimci fonksiyon. counting_model_loss, gorev
    3'teki (bigram_counting.py) ayni veri seti icin bulunan NLL degeri
    karsilastirma yapabilmek icin disaridan veriyoruz.
    """
    print("#" * 60)
    print(f"# {dataset_name}")
    print("#" * 60)

    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    print(f"Alfabe buyuklugu: {vocab_size}")

    xs, ys = build_training_set(words, stoi)
    print(f"Toplam ornek (bigram) sayisi: {xs.shape[0]}")
    print()

    W = init_weights(vocab_size)
    print(f"Agirlik matrisi sekli: {W.shape}")

    probs = forward_pass(xs, W, vocab_size)
    initial_loss = compute_loss(probs, ys, W)
    print(f"Baslangic loss: {initial_loss.item():.4f}")
    print()

    print("Gradient descent ile egitim:")
    W = train(xs, ys, W, vocab_size, steps=200, learning_rate=50.0,
              reg_strength=0.01, verbose_every=40)
    print()

    final_probs = forward_pass(xs, W, vocab_size)
    final_loss = compute_loss(final_probs, ys, W)
    print(f"Sinir agi (NN) final loss   : {final_loss.item():.4f}")
    print(f"Sayma modeli (gorev 3) loss : {counting_model_loss:.4f}")
    print("-> Ikisi birbirine yakinsamis olmali, cunku ayni optimal")
    print("   cozume iki farkli yontemle (analitik vs iteratif) ulasiyoruz.")
    print()

    print("Sinir agi modelinden uretilen ornek isimler:")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(15):
        print(" ", sample_from_nn(W, itos, vocab_size, g))
    print()

    return W

if __name__ == "__main__":
    # --- Ingilizce veri seti ---
    # counting_model_loss degeri, bigram_counting.py'nin names.txt
    # icin urettigi NLL (gorev 3'ten).
    words_en = read_words("names.txt")
    run_bigram_nn_model(words_en, "INGILIZCE ISIM LISTESI (names.txt)",
                         counting_model_loss=2.4544)

    # --- Turkce veri seti ---
    # counting_model_loss degeri, bigram_counting.py'nin
    # turkce_isimler.txt icin urettigi NLL (gorev 3'ten).
    words_tr = read_words("turkce_isimler.txt")
    run_bigram_nn_model(words_tr, "TURKCE ISIM LISTESI (turkce_isimler.txt)",
                         counting_model_loss=2.4575)