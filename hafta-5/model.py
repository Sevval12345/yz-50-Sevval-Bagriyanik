import torch

def init_parameters(vocab_size, embedding_dim, block_size, n_hidden, seed=2147483647):
    """
    Parametreleri Kaiming init mantigiyla baslatiyoruz (gecen haftayla(hafta-3) ayni).

    ONEMLI NOT: BatchNorm kullanilsa bile b1'i BILEREK ekliyoruz, .
    Karpathy'nin egzersiz notebook'unda oldugu gibi: BatchNorm zaten
    her ornekten ortalamayi cikardigi icin b1'in etkisi matematiksel
    olarak SIFIRLANIYOR (gereksiz/"spurious" bir parametre yani). Bunu
    bilerek tutuyoruz cunku gorev 2'de "b1'in gradyaninin sifir
    cikmasi gerektigini" göreceğiz. Bu bize BatchNorm'un
    nasil calistigini cok güzel anlatiyor.
    """
    g = torch.Generator().manual_seed(seed)
    fan_in = block_size * embedding_dim
    gain = 5 / 3

    C = torch.randn((vocab_size, embedding_dim), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g) * (gain / fan_in ** 0.5)
    b1 = torch.randn(n_hidden, generator=g) * 0.1  # kasitli olarak "gereksiz" birakildi
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    total_params = sum(p.nelement() for p in parameters)
    print(f"Toplam parametre sayisi: {total_params}")

    return parameters