import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------
# Transformer
# ------------------------
class SelfiesTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, bos_idx, eos_idx,
                 d_model=192, nhead=6, num_layers=4, dim_feedforward=512,
                 dropout=0.1, max_len=128):
        super(SelfiesTransformer, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.d_model = d_model
        self.max_len = max_len

        # Embeddings
        # تنظیم padding_idx=pad_idx باعث می‌شود گرادیان برای توکن پدینگ محاسبه نشود
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # ساخت لایه‌های انکودر برای ساختار Decoder-Only
        # Decoder has cross and self attention, and we don't need cross attention
        # Encoder has only self attention
        # با استفاده از Causal Mask، این انکودر رفتاری کاملاً مشابه دکودر خواهد داشت.
        # Can only look at the past inputs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN (پایداری بیشتر در آموزش)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz, device):
        # تولید ماسک بالا-مثلثی (Causal Mask)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device

        # ایجاد ماسک‌ها
        causal_mask = self.generate_causal_mask(seq_len, device)
        # ماسک پدینگ (مقادیر True نادیده گرفته می‌شوند)
        padding_mask = (x == self.pad_idx)

        # ایجاد اندیس‌های موقعیت
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # اعمال امبدینگ‌ها
        x_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        x_emb = x_emb + self.position_embedding(positions)
        x_emb = self.dropout(x_emb)

        # عبور از بلاک‌های ترنسفورمر
        out = self.transformer(
            x_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True  # به مدل می‌فهمونه ساختار ما از نوع پیش‌بینی آینده (Autoregressive) است تا محاسبات رو سریع‌تر انجام بده.
        )

        out = self.ln_f(out)
        logits = self.lm_head(out)
        return logits, out

    def get_representation(self, x, eos_positions=None):
        _, hidden_states = self.forward(x)

        if eos_positions is not None:
            eos_positions = eos_positions.long()
            eos_positions = torch.clamp(eos_positions, min=0, max=x.size(1) - 1)
            batch_indices = torch.arange(x.size(0), device=x.device)
            return hidden_states[batch_indices, eos_positions]

        non_pad_mask = (x != self.pad_idx).long()
        last_indices = non_pad_mask.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0, max=x.size(1) - 1)

        batch_indices = torch.arange(x.size(0), device=x.device)
        return hidden_states[batch_indices, last_indices]

# ----------------------
# SELFIES Generation
# ----------------------
def sample_next_token(logits, temperature=1.0, top_k=20, top_p=0.95, forbidden_indices=None):
    logits = logits.clone()
    logits = logits / temperature

    if forbidden_indices is not None and len(forbidden_indices) > 0:
        logits[forbidden_indices] = float("-inf")

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        min_value = values[-1]
        logits[logits < min_value] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # از: [False, True, True, True]
        # به: [False, False, True, True]
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        # مطمئن می‌شود که اولین توکن (محتمل‌ترین گزینه) هیچ‌وقت حذف نشود.
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # Fallback if filtering removes everything
    if not torch.isfinite(logits).any():
        logits = torch.zeros_like(logits)
        if forbidden_indices is not None and len(forbidden_indices) > 0:
            logits[forbidden_indices] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    # sampling randomly
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_selfies(
    model,
    token2idx,
    idx2token,
    max_len=128,
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    num_samples=300, # number of SELFIES generated
):
    model.eval()
    device = next(model.parameters()).device

    pad_idx = model.pad_idx
    bos_idx = model.bos_idx
    eos_idx = model.eos_idx
    unk_idx = token2idx.get("<unk>", None)

    generated_sequences_str = []
    representations = []

    for _ in range(num_samples):
        input_seq = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        for step in range(max_len - 1):
            logits, _ = model(input_seq)
            # we use only one batch for generation each time (300 times) -> one molecule
            # از آخرین position → -1
            # تمام logits مربوط به تمام توکن‌های vocab را بردار → :
            next_logits = logits[0, -1, :]

            forbidden = [pad_idx, bos_idx]
            if unk_idx is not None:
                forbidden.append(unk_idx)

            # Do not allow <eos> in the first 5 generation steps
            if step < 5:
                forbidden.append(eos_idx)

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_indices=forbidden,
            )

            input_seq = torch.cat([input_seq, next_token.view(1, 1)], dim=1)

            if next_token.item() == eos_idx:
                break

        eos_mask = (input_seq == eos_idx)
        if eos_mask.any():
            eos_pos = eos_mask.float().argmax(dim=1).long()
        else:
            # If no EOS, returns last position
            eos_pos = torch.tensor([input_seq.size(1) - 1], dtype=torch.long, device=device)

        rep = model.get_representation(input_seq, eos_positions=eos_pos)
        representations.append(rep.squeeze(0))

        tokens = [idx2token[idx] for idx in input_seq.squeeze(0).tolist()]
        clean_tokens = [t for t in tokens if t not in ["<bos>", "<eos>", "<pad>"]]
        selfies_str = "".join(clean_tokens)
        generated_sequences_str.append(selfies_str)

    return generated_sequences_str, torch.stack(representations, dim=0)


def generate_all_temperatures(transformer, token2idx, idx2token, max_len=128):
    results = {}
    for temp in [0.8, 1.0, 1.2]:
        seqs, reps = generate_selfies(
            model=transformer,
            token2idx=token2idx,
            idx2token=idx2token,
            max_len=max_len,
            temperature=temp,
            top_k=20,
            top_p=0.95,
            num_samples=300,
        )
        results[temp] = {
            "sequences": seqs,
            "representations": reps,
        }
    return results