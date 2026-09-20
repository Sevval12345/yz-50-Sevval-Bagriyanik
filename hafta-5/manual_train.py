import torch
import torch.nn.functional as F
from dataset import read_words, build_vocab, build_dataset, split_words
from model import init_parameters
from backprop import forward_split
from compare import manual_backward, cmp

def verify_simplified_formulas(Xb, Yb, parameters, n):
    """
    Once, kisayol formullerimizin GOREV 2'deki tam zincirle ayni
    sonucu verdigini dogruluyoruz boylece egitime gecmeden once
    formullerin dogru oldugundan emin oluyoruz.
    """
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    for p in parameters:
        p.requires_grad = True

    loss, intermediates = forward_split(Xb, Yb, parameters, n=n)
    for p in parameters:
        p.grad = None
    for t in intermediates.values():
        t.grad = None
    loss.backward()

    # Tam zincirle hesaplanan referans gradyanlar (gorev 2)
    full_grads = manual_backward(Xb, Yb, parameters, intermediates, n=n)

    logits = intermediates['logits']
    h = intermediates['h']
    hpreact = intermediates['hpreact']
    bnraw = intermediates['bnraw']
    bnvar_inv = intermediates['bnvar_inv']

    # --- Kisayol 1: cross entropy'nin tek satirlik gradyani ---
    dlogits_simple = F.softmax(logits, 1)
    dlogits_simple[range(n), Yb] -= 1
    dlogits_simple /= n

    print("Cross entropy kisayolu dogrulamasi:")
    cmp('dlogits (simplified)', dlogits_simple, logits)
    print(f"  (tam zincirle hesaplanan dlogits ile maxdiff: "
          f"{(dlogits_simple - full_grads['logits']).abs().max().item():.2e})")
    print()

    # --- Kisayol 2: BatchNorm'un tek ifadeye inen gradyani ---
    dh_simple = dlogits_simple @ W2.T
    dhpreact_simple = (1.0 - h ** 2) * dh_simple
    dhprebn_simple = bngain * bnvar_inv / n * (
        n * dhpreact_simple
        - dhpreact_simple.sum(0)
        - n / (n - 1) * bnraw * (dhpreact_simple * bnraw).sum(0)
    )

    print("BatchNorm kisayolu dogrulamasi:")
    cmp('dhprebn (simplified)', dhprebn_simple, intermediates['hprebn'])
    print()

    for p in parameters:
        p.requires_grad = False
        p.grad = None

def forward_efficient(Xb, parameters, running_mean, running_var, n, training=True, momentum=0.001, eps=1e-5):
    """
    Verimli forward pass: GOREV 1'deki gibi her adimi ayri
    degiskende tutmuyoruz, sadece backward icin GEREKLI olan
    (h, bnraw, bnvar_inv, embcat, emb, logits) degerleri saklıyoruz.
    """
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1

    if training:
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, unbiased=True, keepdim=True)
        with torch.no_grad():
            running_mean *= (1 - momentum)
            running_mean += momentum * bnmean
            running_var *= (1 - momentum)
            running_var += momentum * bnvar
    else:
        bnmean = running_mean
        bnvar = running_var

    bnvar_inv = (bnvar + eps) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    cache = {'emb': emb, 'embcat': embcat, 'h': h, 'bnraw': bnraw, 'bnvar_inv': bnvar_inv, 'logits': logits}
    return logits, cache

def backward_manual(Xb, Yb, cache, parameters, n):
    """
    GOREV 3: Cross entropy ve BatchNorm'un TEK IFADEYE INDIRGENMIS
    kisayol formulleriyle geriye yayilim: ara degisken YOK, dogrudan
    sonuca atlıyoruz.
    """
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    emb, embcat, h = cache['emb'], cache['embcat'], cache['h']
    bnraw, bnvar_inv, logits = cache['bnraw'], cache['bnvar_inv'], cache['logits']

    # --- Cross entropy kisayolu ---
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    dhpreact = (1.0 - h ** 2) * dh

    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)

    # --- BatchNorm kisayolu ---
    dhprebn = bngain * bnvar_inv / n * (
        n * dhpreact
        - dhpreact.sum(0)
        - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
    )

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    dC.index_add_(0, Xb.view(-1), demb.view(-1, demb.shape[-1]))

    return {'C': dC, 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'bngain': dbngain, 'bnbias': dbnbias}

def train_manual(Xtr, Ytr, Xdev, Ydev, parameters, running_mean, running_var,
                  learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000):
    """
    ADIM: Egitim dongusu: loss.backward() HIC CAGRILMIYOR. Her
    parametrenin gradyani backward_manual() ile elle hesaplaniyor,
    guncelleme de elle yapiliyor.
    """
    param_names = ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias']

    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits, cache = forward_efficient(Xb, parameters, running_mean, running_var, batch_size, training=True)
        loss = F.cross_entropy(logits, Yb)  # sadece IZLEME icin, backward icin degil

        grads = backward_manual(Xb, Yb, cache, parameters, batch_size)

        for name, p in zip(param_names, parameters):
            p.data += -learning_rate * grads[name]

        if k % eval_every == 0 or k == steps - 1:
            with torch.no_grad():
                dev_logits, _ = forward_efficient(Xdev, parameters, running_mean, running_var,
                                                    Xdev.shape[0], training=False)
                dev_loss = F.cross_entropy(dev_logits, Ydev)
            print(f"  adim {k:5d}  train loss = {loss.item():.4f}   dev loss = {dev_loss.item():.4f}")

    return parameters

if __name__ == "__main__":
    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 3, 10, 200
    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size=block_size)

    parameters = init_parameters(vocab_size, embedding_dim, block_size, n_hidden)

    # --- Once kisayol formullerini dogrula ---
    batch_size = 32
    g = torch.Generator().manual_seed(2147483647)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    print("=" * 60)
    print("Kisayol formullerini gorev 2'deki tam zincirle dogrulama")
    print("=" * 60)
    verify_simplified_formulas(Xb, Yb, parameters, n=batch_size)

    # --- Simdi loss.backward() OLMADAN egitim ---
    for p in parameters:
        p.requires_grad = False

    running_mean = torch.zeros((1, n_hidden))
    running_var = torch.ones((1, n_hidden))

    print("=" * 60)
    print("loss.backward() OLMADAN egitim (kendi gradyanlarimizla)")
    print("=" * 60)
    parameters = train_manual(Xtr, Ytr, Xdev, Ydev, parameters, running_mean, running_var,
                               learning_rate=0.1, batch_size=32, steps=20000, eval_every=4000)

    with torch.no_grad():
        final_logits, _ = forward_efficient(Xdev, parameters, running_mean, running_var,
                                              Xdev.shape[0], training=False)
        final_dev_loss = F.cross_entropy(final_logits, Ydev).item()
    print(f"\nAsıl dev loss (loss.backward() hic kullanilmadan): {final_dev_loss:.4f}")