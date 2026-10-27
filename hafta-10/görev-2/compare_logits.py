# Görev 2: Kendi GPT-2'mizin logits'i Hugging Face'inkiyle aynı mı?

import tiktoken
import torch
from transformers import GPT2LMHeadModel
from gpt2 import GPT

torch.manual_seed(1337)

model_hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
model = GPT.from_pretrained("gpt2").eval()
print(f"Kendi modelimizin parametre sayısı (lm_head ayrı sayılıyor): "
      f"{sum(p.numel() for p in model.parameters()):,}")

enc = tiktoken.get_encoding("gpt2")

def karsilastir(ad, idx):
    with torch.no_grad():
        logits_hf = model_hf(idx).logits
        logits_my, _ = model(idx)
    fark = (logits_hf - logits_my).abs().max().item()
    print(f"\n{ad}")
    print(f"  girdi boyutu    : {tuple(idx.shape)}")
    print(f"  logits boyutu   : HF {tuple(logits_hf.shape)}  bizim {tuple(logits_my.shape)}")
    print(f"  en büyük mutlak fark : {fark:.2e}")
    print(f"  allclose(atol=1e-4)  : {torch.allclose(logits_hf, logits_my, atol=1e-4)}")
    print(f"  son pozisyonda en olası token aynı mı: "
          f"{logits_hf[0, -1].argmax().item() == logits_my[0, -1].argmax().item()} "
          f"({enc.decode([logits_my[0, -1].argmax().item()])!r})")

# 1) Gerçek bir cümle
idx = torch.tensor([enc.encode("Hello, I'm a language model,")])
karsilastir("1) Gerçek cümle: \"Hello, I'm a language model,\"", idx)

# 2) Rastgele token'lardan oluşan 2 örnekli batch
idx = torch.randint(0, 50257, (2, 16))
karsilastir("2) Rastgele token'lar (batch=2, T=16)", idx)

"""
- Hafta 8'deki GPT’den farklar neler? -

Mimarinin büyük kısmı aynı kaldı. Hafta 8'deki modelimde de
pre-norm LayerNorm (x + sa(ln1(x))), blokların sonunda ln_f,
residual bağlantılar, tril maskesi, 1/√head_size ölçeklemesi,
öğrenilen pozisyon embedding’i ve dropout vardı. GPT-2'de değişen şunlar:

    1. q, k, v tek katmanda hesaplanıyor.
    Hafta 8'de her head ayrı bir Head modülüydü, her birinin kendi
    key, query, value Linear’ları vardı ve çıktılar torch.cat ile
    birleşiyordu (nn.ModuleList). GPT-2'de tek bir c_attn = Linear(768, 3*768) var.
    Çıktıyı split(768, dim=2) ile q, k, v’ye ayırıp 
    (B, n_head, T, head_size) şekline getiriyorum, yani head’ler
    ayrı modüller yerine tek tensörün bir boyutu oluyor. Hesap aynı,
    ama tek büyük matris çarpımı GPU’da çok daha hızlı.

    2. Aktivasyon ReLU yerine GELU (tanh yaklaşımlı).
    nn.GELU(approximate="tanh"). ReLU negatif tarafı tamamen sıfırlıyor
    (türev 0), GELU yumuşak geçiş yapıyor. Tanh yaklaşımı OpenAI’ın eski
    TensorFlow kodundan kalma. Tam GELU’yla fark çok küçük ama logits’in
    Hugging Face ile birebir eşleşmesi için aynısını kullanmak gerekiyor.

    3. Attention’ı F.scaled_dot_product_attention(q, k, v, is_causal=True) ile yazıyorum.
    Hafta 8'de q @ k.T, masked_fill, softmax ve @ v adımlarını elle yazmıştım.
    Sonuç aynı, ama bu fonksiyon daha hızlu ve daha az bellek kullanıyor.
    Elle yazdığım wei’yi saklayıp ısı haritası çizebilmiştim, bu yöntemde wei görünmüyor.

    4. Bias terimleri farklı.
    Hafta 8'de key/query/value bias’sızdı (bias=False), GPT-2'de
    c_attn’in bias’ı var. Hafta 8'de lm_head nn.Linear varsayılanıyla
    bias’lıydı, GPT-2'de lm_head bias’sız (bias=False).

    5. Dropout’u bu görevde koymadım.
    Hafta 8'de dropout=0.2 kullandım. Hugging Face config’inde
    dropout var ama eval() modunfa kapalı, ve bu görevde sadece
    çıkarım yaptığım için modülüme eklemedim.

    6. Ölçek ve tokenizer.
    Hafta 8'de n_embd=64, 4 head, 3 blok, block_size=32, yaklaşık 160K
    parametre ve karakter tokenizer’ı (vocab 65 veya 103) vardı.
    GPT-2'de 12 head, 12 blok, n_embd=768, block_size=1024, yaklaşık 124M
    parametre ve BPE tokenizer’ı (vocab 50,257) var.

    7. Parametre isimleri.
    Ağırlıkları Hugging Face’ten kopyalayabilmem için modülleri onun
    isimleriyle adlandırdım (transformer.wte, transformer.wpe,
    transformer.h.0.attn.c_attn, mlp.c_fc, ln_f, lm_head ...).
    Hafta 8'de token_embedding_table, blocks, sa, ffwd gibi kendi isimlerimi kullanmıştım.

    
- Ağırlıkları yüklerken neden transpoze ettim? -

OpenAI’ın orijinal TensorFlow kodu dört katmanda Conv1D kullandı ve Hugging Face
bunu Conv1D sınıfı olarak taşıdı. Conv1D ağırlığı (girdi, çıktı) şeklinde saklıyor
ve y = x @ W + b hesaplıyor. nn.Linear ise ağırlığı (çıktı, girdi) şeklinde saklıyor
ve y = x @ W.T + b hesaplıyor. Hesap aynı, saklama düzeni ters. Bu yüzden kopyalarken .t() kullandım.

Transpoze ettiklerim: attn.c_attn.weight, attn.c_proj.weight, mlp.c_fc.weight, mlp.c_proj.weight.
Örnek: Hugging Face’te c_attn.weight (768, 2304), benim nn.Linear(768, 2304).weight’im (2304, 768).

Transpoze etmediklerim:

- wte ve wpe: bunlar matris çarpımı değil, satır seçen tablolar.
- LayerNorm ağırlıkları ve tüm bias’lar: tek boyutlu tensörler.

wte ile lm_head Hugging Face’te aynı bellekte (weight tying). Bu görevde modelimde
ayrı tensörler olarak tutuyorum, ikisine de aynı değerleri kopyalıyorum. Bu yüzden
parametre sayım 163,037,184 çıktı, Hugging Face’in 124,439,808'inden tam
50257 x 768 = 38,597,376 fazla. Weight tying’i Ek (a)’da ekleyeceğim.
"""