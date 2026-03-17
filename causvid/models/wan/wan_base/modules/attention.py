# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _prepare_sdpa_inputs(q, k, v, dtype):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if q.device.type == 'cpu' and dtype in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
    else:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)

    return q, k, v


def _build_length_mask(batch_size, q_len, k_len, device, q_lens, k_lens, causal, window_size):
    mask = torch.ones((batch_size, q_len, k_len), dtype=torch.bool, device=device)

    q_idx = torch.arange(q_len, device=device).view(1, q_len, 1)
    k_idx = torch.arange(k_len, device=device).view(1, 1, k_len)

    if q_lens is not None:
        q_lens = q_lens.to(device=device, dtype=torch.long)
        mask = mask & (q_idx < q_lens.view(batch_size, 1, 1))

    if k_lens is not None:
        k_lens = k_lens.to(device=device, dtype=torch.long)
        mask = mask & (k_idx < k_lens.view(batch_size, 1, 1))

    if causal:
        mask = mask & (k_idx <= q_idx)

    if window_size != (-1, -1):
        left, right = window_size
        if left >= 0:
            mask = mask & (k_idx >= q_idx - left)
        if right >= 0:
            mask = mask & (k_idx <= q_idx + right)

    return mask.unsqueeze(1)


def _merge_sdpa_masks(length_mask, attn_mask, dtype):
    if attn_mask is None:
        return length_mask

    if attn_mask.dtype == torch.bool:
        return length_mask & attn_mask

    additive_mask = torch.zeros_like(length_mask, dtype=dtype)
    additive_mask = additive_mask.masked_fill(~length_mask, float('-inf'))
    return additive_mask + attn_mask.to(dtype)


def _sdpa_attention_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    attn_mask=None,
):
    out_dtype = q.dtype
    batch_size, q_len, k_len = q.size(0), q.size(1), k.size(1)

    q, k, v = _prepare_sdpa_inputs(q, k, v, dtype)

    total_scale = 1.0
    if q_scale is not None:
        total_scale *= q_scale
    if softmax_scale is not None:
        total_scale *= softmax_scale
    if total_scale != 1.0:
        q = q * total_scale

    mask = _build_length_mask(
        batch_size=batch_size,
        q_len=q_len,
        k_len=k_len,
        device=q.device,
        q_lens=q_lens,
        k_lens=k_lens,
        causal=causal,
        window_size=window_size,
    )
    mask = _merge_sdpa_masks(mask, attn_mask, q.dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=False,
        dropout_p=dropout_p,
    )

    if q_lens is not None:
        q_valid = (
            torch.arange(q_len, device=out.device).view(1, q_len, 1)
            < q_lens.to(device=out.device, dtype=torch.long).view(batch_size, 1, 1)
        ).unsqueeze(1)
        out = out.masked_fill(~q_valid, 0)

    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        warnings.warn(
            'flash_attn is not installed; falling back to scaled_dot_product_attention.',
            stacklevel=2,
        )
        return _sdpa_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        return _sdpa_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            attn_mask=attn_mask,
        )
