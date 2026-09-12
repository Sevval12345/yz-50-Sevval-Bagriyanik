import torch
import torch.nn.functional as F
from model import forward_pass

def sample_from_mlp(parameters, stoi, itos, block_size, generator, max_len=30):
    """
    Egitilmis MLP'den bir isim orneklemek. Bos baglamla (block_size
    kadar '.' ile) basliyoruz, modelin tahmin ettigi dagilimdan bir
    karakter ornekliyoruz, baglami kaydirip devam ediyoruz. '.'
    cikana kadar devam ediyoruz.
    """
    C, W1, b1, W2, b2 = parameters
    context = [0] * block_size
    out = []
    for _ in range(max_len):
        X = torch.tensor([context])
        with torch.no_grad():
            logits = forward_pass(X, C, W1, b1, W2, b2)
            probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return ''.join(out)

def bigram_sample_name(P, itos, generator):
    """Bigram modelinden isim orneklemek gecen haftaki (hafta-3) mantigin aynisi."""
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break
    return ''.join(out)

def build_bigram_baseline(words, stoi, vocab_size):
    """
    MLP ile karsilastirma icin basit bigram baseline modeli. Gecen
    haftaki sayma tabanli bigram modelinin kucuk, tek dosyalik hali.
    """
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    P = (N + 1).float()
    P = P / P.sum(1, keepdim=True)
    return P