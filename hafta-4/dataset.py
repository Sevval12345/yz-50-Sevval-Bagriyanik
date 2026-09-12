import random
import torch

def read_words(path="names.txt"):
    """
    karpathy'nin names.txt dosyasini okuyup her satiri bir isim olarak bir listeye
    topluyoruz.
    """
    with open(path, "r") as f:
        words = f.read().splitlines()
    return words

def build_vocab(words):
    """
    stoi (string to integer) ve itos (integer to string) sozluklerini
    kuruyor. '.' karakteri hem kelime baslangici hem bitisi anlamina
    geliyor, ona ozel olarak 0 indeksini ayiriyoruz (hafta 3'le aynı gidiyoruz).
    """
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def build_dataset(words, stoi, block_size=3):
    """
    X (baglam) ve Y (hedef) veri setini kuruyoruz. block_size: kac
    onceki harfe bakacagimiz demek. Her kelimenin basina 'block_size' kadar
    '.' ekliyoruz, sonra kelimeyi harf harf gezip baglami bir harf
    kaydirarak ilerliyoruz.
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def split_words(words, train_frac=0.8, dev_frac=0.1, seed=42):
    """
    Veriyi train/dev/test olarak %80/%10/%10 boluyoruz. Once kelimeleri
    KARISTIRIYORUZ, cunku names.txt alfabetik sirali olabilir (ama değil. bunu önlem olarak yapıyoruz).
    """
    words = words[:]
    random.Random(seed).shuffle(words)
    n = len(words)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train_words = words[:n_train]
    dev_words = words[n_train:n_train + n_dev]
    test_words = words[n_train + n_dev:]
    return train_words, dev_words, test_words