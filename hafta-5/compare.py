import torch
import torch.nn.functional as F
from dataset import read_words, build_vocab, build_dataset, split_words
from model import init_parameters
from backprop import forward_split

def cmp(s, dt, t):
    """
    Elle hesapladigimiz gradyan (dt) ile PyTorch'un retain_grad() ile
    sakladigi gercek gradyani (t.grad) karsilastiriyoruz.
    - exact: birebir ayni mi? (cok siki bir esitlik)
    - approximate: floating-point toleransiyla yakin m? (atol=1e-5)
    - maxdiff: en buyuk mutlak fark: hangi yonde/ne kadar saptigimizi gosterir
    """
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad, atol=1e-5)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

def manual_backward(Xb, Yb, parameters, intermediates, n):
    """
    GOREV 2: forward_split'teki her adimin gradyanini elle, zincir
    kuralini geriye dogru uygulayarak hesapliyoruz. Yorumlar, her
    adimda HANGI ileri-yon islemin tersini aldigimizi aciklyor.
    """
    C, W1, b1, W2, b2, bngain, bnbias = parameters
    emb = intermediates['emb']
    embcat = intermediates['embcat']
    hprebn = intermediates['hprebn']
    bnmeani = intermediates['bnmeani']
    bndiff = intermediates['bndiff']
    bndiff2 = intermediates['bndiff2']
    bnvar = intermediates['bnvar']
    bnvar_inv = intermediates['bnvar_inv']
    bnraw = intermediates['bnraw']
    hpreact = intermediates['hpreact']
    h = intermediates['h']
    logits = intermediates['logits']
    logit_maxes = intermediates['logit_maxes']
    norm_logits = intermediates['norm_logits']
    counts = intermediates['counts']
    counts_sum = intermediates['counts_sum']
    counts_sum_inv = intermediates['counts_sum_inv']
    probs = intermediates['probs']
    logprobs = intermediates['logprobs']

    grads = {}

    # loss = -logprobs[range(n), Yb].mean()
    # Sadece DOGRU karakterin logprob'u loss'u etkiliyor, digerlerinin gradyani sifir. mean() oldugu icin 1/n ile bolunuyor.
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0 / n
    grads['logprobs'] = dlogprobs

    # logprobs = probs.log()  ->  d/dprobs log(probs) = 1/probs
    dprobs = (1.0 / probs) * dlogprobs
    grads['probs'] = dprobs

    # probs = counts * counts_sum_inv
    # counts_sum_inv BROADCAST ediliyor (tek sutun -> tum sutunlar).
    # Geriye donerken: broadcast edilenin gradyani SUM ile geri doner.
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    grads['counts_sum_inv'] = dcounts_sum_inv

    # counts_sum_inv = counts_sum ** -1  ->  d/dx (x^-1) = -x^-2
    dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
    grads['counts_sum'] = dcounts_sum

    # counts iki yoldan kullaniliyor - ikisinin de katkisini TOPLUYORUZ:
    # 1) probs = counts * counts_sum_inv (dogrudan carpim)
    dcounts = counts_sum_inv * dprobs
    # 2) counts_sum = counts.sum(1, keepdim=True) (SUM -> geri BROADCAST)
    dcounts = dcounts + torch.ones_like(counts) * dcounts_sum
    grads['counts'] = dcounts

    # counts = norm_logits.exp()  ->  d/dx exp(x) = exp(x) = counts
    dnorm_logits = counts * dcounts
    grads['norm_logits'] = dnorm_logits

    # norm_logits = logits - logit_maxes
    # logits iki yoldan geliyor - burada ilk katki (dogrudan fark):
    dlogits = dnorm_logits.clone()
    # logit_maxes'in gradyani: -1 carpani ile, ve SUM (broadcast'in tersi)
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    grads['logit_maxes'] = dlogit_maxes

    # logit_maxes = logits.max(1, keepdim=True).values
    # Sadece MAKSIMUM olan pozisyona gradyan akar (digerleri max'i
    # etkilemiyor). one_hot ile "kim maksimumdu" maskesini olusturup
    # gradyani sadece o pozisyona ekliyoruz (ikinci katki, topluyoruz).
    dlogits = dlogits + F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
    grads['logits'] = dlogits

    # logits = h @ W2 + b2
    # Matris carpiminin gradyani: dh = dlogits @ W2.T, dW2 = h.T @ dlogits
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    grads['h'] = dh
    grads['W2'] = dW2
    grads['b2'] = db2

    # h = tanh(hpreact)  ->  d/dx tanh(x) = 1 - tanh(x)^2 = 1 - h^2
    dhpreact = (1.0 - h ** 2) * dh
    grads['hpreact'] = dhpreact

    # hpreact = bngain * bnraw + bnbias
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    grads['bngain'] = dbngain
    grads['bnraw'] = dbnraw
    grads['bnbias'] = dbnbias

    # bnraw = bndiff * bnvar_inv - bndiff iki yoldan gelecek, ilk katki:
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    grads['bnvar_inv'] = dbnvar_inv

    # bnvar_inv = (bnvar + 1e-5) ** -0.5  ->  d/dx x^-0.5 = -0.5 * x^-1.5
    dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
    grads['bnvar'] = dbnvar

    # bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
    # SUM -> geri BROADCAST, ve 1/(n-1) carpani (Bessel duzeltmesi)
    dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
    grads['bndiff2'] = dbndiff2

    # bndiff2 = bndiff ** 2  ->  d/dx x^2 = 2x  (ikinci katki, TOPLUYORUZ)
    dbndiff = dbndiff + (2 * bndiff) * dbndiff2
    grads['bndiff'] = dbndiff

    # bndiff = hprebn - bnmeani  -  hprebn iki yoldan gelecek, ilk katki:
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0, keepdim=True)
    grads['bnmeani'] = dbnmeani

    # bnmeani = 1/n * hprebn.sum(0, keepdim=True)
    # SUM -> geri BROADCAST, 1/n carpani (ikinci katki, TOPLUYORUZ)
    dhprebn = dhprebn + (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
    grads['hprebn'] = dhprebn

    # hprebn = embcat @ W1 + b1
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    grads['embcat'] = dembcat
    grads['W1'] = dW1
    grads['b1'] = db1

    # embcat = emb.view(emb.shape[0], -1)  ->  view'in gradyani, yani ayni sayida elemani orijinal sekle geri koymak (reshape'in tersi)
    demb = dembcat.view(emb.shape)
    grads['emb'] = demb

    # emb = C[Xb]  -  bu bir INDEKSLEME islemi. C'nin her satiri, Xb'de KAC KEZ kullanildiysa o kadar katki alir bu yuzden index_add_ (ya da esdegeri bir dongu) ile TOPLAYARAK biriktiriyoruz
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k, j]
            dC[ix] += demb[k, j]
    grads['C'] = dC

    return grads

if __name__ == "__main__":
    words = read_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    block_size, embedding_dim, n_hidden = 3, 10, 200
    train_words, dev_words, test_words = split_words(words)
    Xtr, Ytr = build_dataset(train_words, stoi, block_size=block_size)

    parameters = init_parameters(vocab_size, embedding_dim, block_size, n_hidden)

    batch_size = 32
    g = torch.Generator().manual_seed(2147483647)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    loss, intermediates = forward_split(Xb, Yb, parameters, n=batch_size)

    for p in parameters:
        p.grad = None
    for t in intermediates.values():
        t.grad = None
    loss.backward()

    # --- Elle gradyanlari hesapla ---
    grads = manual_backward(Xb, Yb, parameters, intermediates, n=batch_size)

    print("=" * 70)
    print("ARA DEGISKENLERIN KARSILASTIRMASI")
    print("=" * 70)
    ara_degiskenler = [
        'logprobs', 'probs', 'counts_sum_inv', 'counts_sum', 'counts',
        'norm_logits', 'logit_maxes', 'logits', 'h', 'hpreact', 'bnraw',
        'bnvar_inv', 'bnvar', 'bndiff2', 'bndiff', 'bnmeani', 'hprebn',
        'embcat', 'emb',
    ]
    for name in ara_degiskenler:
        cmp(name, grads[name], intermediates[name])

    print()
    print("=" * 70)
    print("PARAMETRELERIN KARSILASTIRMASI")
    print("=" * 70)
    param_names = ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias']
    for name, p in zip(param_names, parameters):
        cmp(name, grads[name], p)