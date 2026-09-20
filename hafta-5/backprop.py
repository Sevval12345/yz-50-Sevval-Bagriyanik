import torch
import torch.nn.functional as F

def forward_split(Xb, Yb, parameters, n):
    """
    GOREV 1: Forward pass'i, gecen hafta (hafta 4) tek satirlik fonksiyonlarla
    (F.cross_entropy, forward_pass_bn) yaptigimiz her seyi, ATOMIK
    PyTorch islemlerine bolerek yeniden yaziyoruz. Her ara sonuc ayri
    bir degiskende, boylece her birinin gradyanini ayri ayri
    inceleyebiliyoruz.

    n: batch'teki ornek sayisi (bazi normalize islemlerinde kullanacagiz)
    """
    C, W1, b1, W2, b2, bngain, bnbias = parameters

    # --- Embedding ---
    emb = C[Xb]                          # embed
    embcat = emb.view(emb.shape[0], -1)  # concatenate (duzlestir)

    # --- Katman 1 (Linear) ---
    hprebn = embcat @ W1 + b1

    # --- BatchNorm (ATOMIK ADIMLARA BOLUNMUS) ---
    bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = 1 / (n - 1) * bndiff2.sum(0, keepdim=True)  # Bessel duzeltmesi: n-1
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # --- Non-linearity ---
    h = torch.tanh(hpreact)

    # --- Katman 2 (cikis) ---
    logits = h @ W2 + b2

    # --- Cross entropy (ATOMIK ADIMLARA BOLUNMUS) ---
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes  # sayisal stabilite icin
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    # Butun ara degiskenleri bir sozlukte topluyoruz, hem retain_grad cagirmak hem de daha sonra PyTorch'un gradyanlarini okumak icin 
    intermediates = {
        'emb': emb, 'embcat': embcat, 'hprebn': hprebn,
        'bnmeani': bnmeani, 'bndiff': bndiff, 'bndiff2': bndiff2,
        'bnvar': bnvar, 'bnvar_inv': bnvar_inv, 'bnraw': bnraw,
        'hpreact': hpreact, 'h': h, 'logits': logits,
        'logit_maxes': logit_maxes, 'norm_logits': norm_logits,
        'counts': counts, 'counts_sum': counts_sum,
        'counts_sum_inv': counts_sum_inv, 'probs': probs,
        'logprobs': logprobs,
    }

    # ADIM: her ara degiskene retain_grad() cagiriyoruz. Bunlar LEAF
    # tensor degil (hesaplanmislar), bu yuzden varsayilan olarak
    # backward() sonrasi .grad'lari atilir. retain_grad() ile PyTorch'a
    # "bunu da sakla" diyoruz - boylece backward()'dan sonra her
    # birinin .grad'ini okuyabiliyoruz.
    for name, t in intermediates.items():
        t.retain_grad()

    return loss, intermediates

if __name__ == "__main__":
    from dataset import read_words, build_vocab, build_dataset, split_words
    from model import init_parameters

    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 3, 10, 200
    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)

    parameters = init_parameters(vocab_size, embedding_dim, block_size, n_hidden)

    # Videodaki gibi kucuk bir minibatch aliyoruz  
    batch_size = 32
    g = torch.Generator().manual_seed(2147483647)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    loss, intermediates = forward_split(Xb, Yb, parameters, n=batch_size)
    print(f"\nLoss: {loss.item():.4f}")

    # --- loss.backward() ile TUM gradyanlari (parametreler + aradegiskenler) tek seferde hesapliyoruz ---
    for p in parameters:
        p.grad = None
    for t in intermediates.values():
        t.grad = None
    loss.backward()

    print("\nPyTorch'un hesapladigi ara degisken gradyanlari (ilk birkaci):")
    for name in ['logprobs', 'probs', 'counts', 'norm_logits', 'logits', 'h', 'hpreact']:
        t = intermediates[name]
        print(f"  {name:<15} sekli: {str(list(t.shape)):<15}  grad sekli: {str(list(t.grad.shape)):<15}  grad var mi: {t.grad is not None}")

    print("\nParametrelerin gradyanlari:")
    param_names = ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias']
    for name, p in zip(param_names, parameters):
        print(f"  {name:<10} sekli: {str(list(p.shape)):<15}  grad var mi: {p.grad is not None}")