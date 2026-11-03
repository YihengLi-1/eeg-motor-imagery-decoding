"""PyTorch's fused scaled_dot_product_attention has no dropout on Apple MPS; fall back to the explicit math there.
Import before building models (braindecode calls F.scaled_dot_product_attention at run time)."""
import torch
import torch.nn.functional as F

_fused = F.scaled_dot_product_attention


def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if q.device.type != "mps" or dropout_p == 0.0:
        return _fused(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    a = (q @ k.transpose(-2, -1)) * (scale if scale is not None else q.size(-1) ** -0.5)
    if is_causal:
        L, S = q.size(-2), k.size(-2)
        a = a.masked_fill(torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    if attn_mask is not None:
        a = a.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else a + attn_mask
    return F.dropout(a.softmax(-1), dropout_p, training=True) @ v


F.scaled_dot_product_attention = _sdpa
