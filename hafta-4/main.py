import math
import torch
import torch.nn.functional as F
from dataset import read_words, build_vocab, build_dataset, split_words
from model import (
    init_layers_naive, init_layers_kaiming, init_layers_batchnorm,
    forward_pass, diagnostic_forward, forward_pass_bn,
)
from train import overfit_single_minibatch, search_learning_rate, train, train_bn, train_with_lr_decay
from sample import sample_from_mlp, bigram_sample_name, build_bigram_baseline
from diagnostics import analyze_saturation, plot_histogram, visualize_embeddings, compare_capacities

BLOCK_SIZE = 3
EMBEDDING_DIM = 2
N_HIDDEN = 100

def gorev_1(words, stoi, itos, vocab_size):
    """
    GOREV 1: 3 harf baglamli veri seti (X, Y) ve embedding tablosu.
    """
    print("=" * 60)
    print("GOREV 1: Veri seti (X, Y) ve embedding tablosu")
    print("=" * 60)

    X, Y = build_dataset(words, stoi, block_size=BLOCK_SIZE)
    print(f"X sekli: {X.shape}   Y sekli: {Y.shape}")

    print("\nIlk 5 (baglam -> hedef) ornegi:")
    for i in range(5):
        baglam = ''.join(itos[ix.item()] for ix in X[i])
        print(f"  {baglam!r:>10} -> {itos[Y[i].item()]!r}")

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, EMBEDDING_DIM), generator=g)
    emb = C[X]
    print(f"\nC (embedding tablosu) sekli: {C.shape}")
    print(f"C[X] sekli: {emb.shape}")
    print()

def gorev_3(Xtr, Ytr, Xdev, Ydev, vocab_size):
    """
    GOREV 2+3: Gizli/cikis katmani, elle loss vs F.cross_entropy (gorev 2),
    tek minibatch overfit, learning rate tarama, tam egitim (gorev 3).
    """
    print("=" * 60)
    print("GOREV 2: Gizli/cikis katmani, elle loss vs F.cross_entropy")
    print("=" * 60)

    parameters = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    C, W1, b1, W2, b2 = parameters
    logits = forward_pass(Xtr[:1000], C, W1, b1, W2, b2)

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss_manual = -probs[torch.arange(1000), Ytr[:1000]].log().mean()
    loss_builtin = F.cross_entropy(logits, Ytr[:1000])
    print(f"Elle hesaplanan loss: {loss_manual.item():.4f}")
    print(f"F.cross_entropy loss : {loss_builtin.item():.4f}")
    print(f"Ikisi ayni mi? {torch.allclose(loss_manual, loss_builtin, atol=1e-4)}")
    print()

    print("=" * 60)
    print("GOREV 3: Overfit testi + LR tarama + tam egitim")
    print("=" * 60)

    parameters = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    overfit_single_minibatch(Xtr, Ytr, parameters, batch_size=32, steps=1000, lr=0.1)
    print()

    parameters = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    best_lr = search_learning_rate(Xtr, Ytr, parameters, batch_size=32, steps=1000)
    print()

    parameters = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    parameters = train(Xtr, Ytr, Xdev, Ydev, parameters,
                        learning_rate=best_lr, batch_size=32, steps=20000, eval_every=4000)

    C, W1, b1, W2, b2 = parameters
    with torch.no_grad():
        dev_loss = F.cross_entropy(forward_pass(Xdev, C, W1, b1, W2, b2), Ydev).item()
    print(f"\nAsıl dev loss: {dev_loss:.4f}\n")
    return parameters

def gorev_4(Xtr, Ytr, Xdev, Ydev, vocab_size, itos, words, stoi):
    """
    GOREV 4: Kapasite karsilastirmasi, embedding gorsellestirme, ornekleme.
    """
    print("=" * 60)
    print("GOREV 4: Kapasite karsilastirmasi + gorsellestirme + ornekleme")
    print("=" * 60)

    configs = [(2, 100), (2, 300), (10, 200), (10, 300)]
    compare_capacities(Xtr, Ytr, Xdev, Ydev, vocab_size, BLOCK_SIZE, configs, steps=15000)
    print()

    parameters = init_layers_kaiming(vocab_size, embedding_dim=2, block_size=BLOCK_SIZE, n_hidden=300)
    parameters = train(Xtr, Ytr, Xdev, Ydev, parameters, learning_rate=0.1, steps=20000, eval_every=5000)
    visualize_embeddings(parameters[0], itos)
    print()

    print("MLP'den uretilen ornek isimler:")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(10):
        print(" ", sample_from_mlp(parameters, stoi, itos, BLOCK_SIZE, g))

    print("\nBigram vs MLP karsilastirmasi:")
    P_bigram = build_bigram_baseline(words, stoi, vocab_size)
    g_bigram = torch.Generator().manual_seed(2147483647)
    g_mlp = torch.Generator().manual_seed(2147483647)
    for _ in range(5):
        print(f"  bigram: {bigram_sample_name(P_bigram, itos, g_bigram):<20} mlp: {sample_from_mlp(parameters, stoi, itos, BLOCK_SIZE, g_mlp)}")
    print()

def gorev_5(Xtr, Ytr, vocab_size):
    """
    GOREV 5: Baslangic loss sorunu, tanh doymasi, Kaiming init ile duzeltme.
    """
    print("=" * 60)
    print("GOREV 5: Baslangic loss + tanh doymasi + Kaiming init")
    print("=" * 60)

    ideal_loss = -math.log(1 / vocab_size)
    print(f"Ideal baslangic loss: {ideal_loss:.4f}\n")

    Xb, Yb = Xtr[:1000], Ytr[:1000]

    params_naive = init_layers_naive(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    logits_naive, h_naive, hpreact_naive = diagnostic_forward(Xb, params_naive)
    loss_naive = F.cross_entropy(logits_naive, Yb)
    print(f"[Naif init] baslangic loss: {loss_naive.item():.4f}")
    analyze_saturation(h_naive, "naif")
    plot_histogram(h_naive, "tanh Ciktilari - Yavas Init", "tanh_hist_naive.png")
    print()

    params_kaiming = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    logits_kaiming, h_kaiming, hpreact_kaiming = diagnostic_forward(Xb, params_kaiming)
    loss_kaiming = F.cross_entropy(logits_kaiming, Yb)
    print(f"[Kaiming init] baslangic loss: {loss_kaiming.item():.4f}")
    analyze_saturation(h_kaiming, "kaiming")
    plot_histogram(h_kaiming, "tanh Ciktilari - Kaiming Init", "tanh_hist_kaiming.png")
    print()

def gorev_6(Xtr, Ytr, Xdev, Ydev, vocab_size):
    """
    GOREV 6: BatchNorm ekleme, BN'li vs BN'siz karsilastirma.
    """
    print("=" * 60)
    print("GOREV 6: BatchNorm: egitimde batch istatistigi, tahminde running mean")
    print("=" * 60)

    print("BatchNorm'suz (Kaiming init):")
    params_no_bn = init_layers_kaiming(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    params_no_bn = train(Xtr, Ytr, Xdev, Ydev, params_no_bn, learning_rate=0.1, steps=20000, eval_every=4000)
    C, W1, b1, W2, b2 = params_no_bn
    with torch.no_grad():
        dev_loss_no_bn = F.cross_entropy(forward_pass(Xdev, C, W1, b1, W2, b2), Ydev).item()
    print()

    print("BatchNorm'lu:")
    params_bn, running_mean, running_std = init_layers_batchnorm(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    params_bn, running_mean, running_std = train_bn(
        Xtr, Ytr, Xdev, Ydev, params_bn, running_mean, running_std,
        learning_rate=0.1, steps=20000, eval_every=4000)
    with torch.no_grad():
        dev_loss_bn = F.cross_entropy(
            forward_pass_bn(Xdev, params_bn, running_mean, running_std, training=False), Ydev).item()

    print(f"\nBatchNorm'suz dev loss: {dev_loss_no_bn:.4f}")
    print(f"BatchNorm'lu  dev loss: {dev_loss_bn:.4f}\n")

def gorev_7():
    """
    GOREV 7: Turkce veriyle ayni modeli egitme, bigram ile karsilastirma.
    """
    print("=" * 60)
    print("GOREV 7: Turkce veriyle egitim")
    print("=" * 60)

    words_tr = read_words("turkce_isimler.txt")
    stoi_tr, itos_tr = build_vocab(words_tr)
    vocab_size_tr = len(stoi_tr)
    print(f"Turkce alfabe buyuklugu: {vocab_size_tr}")

    train_words, dev_words, test_words = split_words(words_tr)
    Xtr, Ytr = build_dataset(train_words, stoi_tr, block_size=BLOCK_SIZE)
    Xdev, Ydev = build_dataset(dev_words, stoi_tr, block_size=BLOCK_SIZE)

    parameters = init_layers_kaiming(vocab_size_tr, EMBEDDING_DIM, BLOCK_SIZE, N_HIDDEN)
    parameters = train(Xtr, Ytr, Xdev, Ydev, parameters, learning_rate=0.1, steps=20000, eval_every=4000)

    C, W1, b1, W2, b2 = parameters
    with torch.no_grad():
        dev_loss = F.cross_entropy(forward_pass(Xdev, C, W1, b1, W2, b2), Ydev).item()
    print(f"\nAsıl dev loss (MLP, Turkce): {dev_loss:.4f}")

    print("\nMLP'den uretilen Turkce ornek isimler:")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(10):
        print(" ", sample_from_mlp(parameters, stoi_tr, itos_tr, BLOCK_SIZE, g))
    print()

def gorev_8():
    """
    GOREV 8 (ek): Hiperparametreleri ayarlayip 2.2 dev loss hedefini gecme.
    """
    print("=" * 60)
    print("GOREV 8 (ek): 2.2 dev loss hedefini gecme")
    print("=" * 60)

    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=BLOCK_SIZE)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=BLOCK_SIZE)

    parameters = init_layers_kaiming(vocab_size, embedding_dim=10, block_size=BLOCK_SIZE, n_hidden=200)
    parameters = train_with_lr_decay(Xtr, Ytr, Xdev, Ydev, parameters,
                                       steps_phase1=40000, lr_phase1=0.1,
                                       steps_phase2=20000, lr_phase2=0.01)

    C, W1, b1, W2, b2 = parameters
    with torch.no_grad():
        dev_loss = F.cross_entropy(forward_pass(Xdev, C, W1, b1, W2, b2), Ydev).item()

    print(f"\nAsıl dev loss: {dev_loss:.4f}  (hedef: 2.2)")
    print("BASARILI" if dev_loss < 2.2 else "HENUZ DEGIL")


if __name__ == "__main__":
    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=BLOCK_SIZE)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=BLOCK_SIZE)

    gorev_1(words, stoi, itos, vocab_size)
    gorev_3(Xtr, Ytr, Xdev, Ydev, vocab_size)
    gorev_4(Xtr, Ytr, Xdev, Ydev, vocab_size, itos, words, stoi)
    gorev_5(Xtr, Ytr, vocab_size)
    gorev_6(Xtr, Ytr, Xdev, Ydev, vocab_size)
    gorev_7()
    gorev_8()