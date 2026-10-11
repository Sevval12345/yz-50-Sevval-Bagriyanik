import torch

def load_text(path="input.txt"):
    """Tiny Shakespeare metnini okuyoruz."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def build_vocab(text):
    """
    Karakter tokenizer'i kuruyoruz. Mantik gecen haftalarin stoi/itos'uyla
    ayni. GPT baglaminda buna 'tokenizer' denir, ama biz karakter
    bazli gidiyoruz (BPE gibi kelime parcasi tabanli degil): her
    benzersiz karakter tek bir token.
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size

def encode(s, stoi):
    """String -> tam sayi listesi."""
    return [stoi[c] for c in s]

def decode(ids, itos):
    """Tam sayi listesi (veya tensoru) -> string."""
    return ''.join([itos[int(i)] for i in ids])

def split_data(data, train_frac=0.9):
    """
    %90 train, %10 val. Gecen haftalarin train/dev/test'inden farkli
    olarak burada sadece iki parca var, cunku artik hiperparametre
    secmek icin ayri bir dev set'e ihtiyacimiz yok. Sadece "model
    genelliyor mu?" sorusuna bakiyoruz.

    Önemli bir sey: words listesini karistirdigimiz gibi karistirmiyoruz. Metin
    sirali bir akis (Shakespeare'in oyunlari bastan sona), karistirirsak
    baglami parcalariz. Bu yuzden basit bolme: ilk %90 train, son %10 val.
    """
    n = int(train_frac * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(data, block_size, batch_size, device="cpu"):
    """
    Gecen haftalarda her ornek bir isimdi (build_dataset kelime kelime
    geziyordu). Burada tek bir uzun metin var, "kelime" kavrami yok.
    Onun yerine metnin icinden batch_size kadar RASTGELE baslangic
    noktasi seciyoruz, her birşnden block_size uzunlugunda bir parca
    kesiyoruz (x), ve x'in bir pozisyon kaydirilmis hali (y) hedef
    oluyor yani x[t]'den sonra gelmesi gereken karakter y[t].

    Bu tek parca aslinda block_size kadar ic ice gecmis bigram/trigram/...
    ornegi uretiyor: pozisyon 0'daki tahmşn sadece 1 onceki karaktere
    bakabilir, pozisyon block_size-1'deki tahmin tum block_size karaktere
    bakabilir (ileride attention bunu kullanacak. Simdilik bigram model
    hepsini ayni sekilde, sadece son karaktere bakarak kullanacak).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y