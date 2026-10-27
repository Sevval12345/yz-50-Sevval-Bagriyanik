# Görev 2: GPT-2 (124M) modülü + Hugging Face ağırlıklarını yükleme.
# Parametre isimleri HF ile birebir aynı: böylece state_dict anahtarları eşleşir.

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024      # maksimum bağlam uzunluğu
    vocab_size: int = 50257     # 50000 BPE birleştirme + 256 byte + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    """12 head'in hepsi tek tensörde: q, k, v tek Linear'dan çıkıyor."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # q, k, v birlikte
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # head'leri birleştirdikten sonra
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)                 # her biri (B, T, C)
        # (B, T, C) -> (B, n_head, T, head_size): head'ler batch gibi davranıyor
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # tril maskesi + softmax + @v
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # head'leri yan yans birleştir
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")                  # GPT-2'nin kullandığı yaklaşık GELU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # pre-norm + residual
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),     # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),     # pozisyon embedding (öğrenilen)
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                       # son LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"T={T} block_size={self.config.block_size}'dan büyük"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)        # (B,T,C) + (T,C) -> broadcast
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                         # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """Hugging Face'in GPT-2 (124M) ağırlıklarını kendi modelimize kopyalar."""
        from transformers import GPT2LMHeadModel

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        model = cls(GPTConfig())
        sd = model.state_dict()

        # Eski HF sürümlerinde maske buffer'ları da state_dict'te çıkar; bunlar öğrenilen ağırlık değil.
        keys_hf = [k for k in sd_hf
                   if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]
        assert len(keys_hf) == len(sd), f"anahtar sayısı farklı: HF {len(keys_hf)} vs bizim {len(sd)}"

        # HF bu 4 ağırlıkta Conv1D kullanıyor: (girdi, çıktı). nn.Linear: (çıktı, girdi) -> transpoze.
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

        with torch.no_grad():
            for k in keys_hf:
                if any(k.endswith(w) for w in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape, f"{k}: {sd_hf[k].shape} vs {sd[k].shape}"
                    sd[k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k].shape, f"{k}: {sd_hf[k].shape} vs {sd[k].shape}"
                    sd[k].copy_(sd_hf[k])
        return model