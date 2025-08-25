# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d
from ..modules.attention import attention

import torch.distributed as dist
import math
import time


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) else start_frame

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    kv_cache: dict = None,
    crossattn_cache: dict = None,
    current_start: int = 0,
    current_end: int = 0
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(x)
    """
    torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])
    """

    # time embeddings
    # with amp.autocast(dtype=torch.float32):
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
    e0 = self.time_projection(e).unflatten(
        1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
    # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=self.block_mask
    )

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block_index, block in enumerate(self.blocks):
        kwargs.update(
            {
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "current_end": current_end
            }
        )
        x = block(x, **kwargs)
    # head
    x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     block_mask,
                     kv_cache=None,
                     current_start=0,
                     current_end=0,
                     ):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    frame_seqlen = frame_seqlen // dist.get_world_size()
    current_start_frame = current_start // frame_seqlen
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(
        k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = self.sink_size * frame_seqlen
    # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]
    if (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

    x = xFuserLongContextAttention()(
        None,
        roped_query,
        kv_cache["k"][:, :local_end_index],
        kv_cache["v"][:, :local_end_index],
        window_size=self.window_size
    )

    # x = attention(
    #     roped_query,
    #     kv_cache["k"][:, :local_end_index],
    #     kv_cache["v"][:, :local_end_index]
    # )
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
