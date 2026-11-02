"""Full-context FP32 Qwen encoding with bounded SDPA query workspaces.

Reuses the immutable v1 job and output contract. Only query rows are tiled:
every row attends to the same full K/V with its original global causal mask.
This is not text chunking, sliding-window attention, or embedding averaging.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import time

import run_kaggle_semantic_encoding as base

VERSION = 'full-context-query-tiled-sdpa-v2'


def tiled_sdpa(original, query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, *, scale=None, enable_gqa=False, chunk_size=256):
    """Inference-only SDPA equivalent with global-position causal masking."""
    import torch
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('Positive query chunk size required')
    length = query.shape[-2]
    if length <= chunk_size:
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if dropout_p != 0:
        raise ValueError('Tiled frozen inference does not support dropout')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)):
        raise ValueError('This tiled implementation is for frozen inference only')
    # Expand grouped K/V once. It is exactly the repeat used by math SDPA GQA,
    # not a change to attention heads, parameters, or sequence coverage.
    if enable_gqa:
        qheads, kheads, vheads = query.shape[-3], key.shape[-3], value.shape[-3]
        if kheads != vheads or qheads % kheads:
            raise ValueError('Invalid grouped-query attention head dimensions')
        key = key.repeat_interleave(qheads // kheads, dim=-3)
        value = value.repeat_interleave(qheads // kheads, dim=-3)
    key_length = key.shape[-2]
    output = torch.empty((*query.shape[:-1], value.shape[-1]), device=query.device, dtype=query.dtype)
    key_positions = torch.arange(key_length, device=query.device) if is_causal else None
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        mask = attn_mask
        if mask is not None and mask.ndim >= 2:
            if mask.shape[-2] == length:
                mask = mask[..., start:end, :]
            elif mask.shape[-2] != 1:
                raise ValueError('Attention mask query dimension cannot be broadcast')
        if is_causal:
            # SDPA causal=True would incorrectly reset the diagonal for each
            # non-square local-Q/full-K call. Use the original global positions.
            causal = key_positions.unsqueeze(0) <= torch.arange(start, end, device=query.device).unsqueeze(1)
            if mask is None:
                mask = causal
            elif mask.dtype == torch.bool:
                mask = mask & causal
            else:
                bias = torch.zeros((end-start, key_length), dtype=query.dtype, device=query.device)
                bias.masked_fill_(~causal, float('-inf'))
                mask = mask + bias
        output[..., start:end, :] = original(
            query[..., start:end, :], key, value, attn_mask=mask,
            dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False)
    return output


@contextmanager
def bounded_query_attention(chunk_size=256):
    """Scoped to this single-process runner; always restores PyTorch's function."""
    import torch.nn.functional as functional
    original = functional.scaled_dot_product_attention
    if getattr(original, '_footballmind_tiled_sdpa', False):
        raise RuntimeError('Nested attention patch is not supported')
    counters = {'attention_calls': 0, 'tiled_attention_calls': 0, 'maximum_query_tokens': 0}
    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        counters['attention_calls'] += 1
        counters['tiled_attention_calls'] += int(query.shape[-2] > chunk_size)
        counters['maximum_query_tokens'] = max(counters['maximum_query_tokens'], query.shape[-2])
        return tiled_sdpa(original, query, key, value, attn_mask, dropout_p, is_causal,
                          scale=scale, enable_gqa=enable_gqa, chunk_size=chunk_size)
    wrapped._footballmind_tiled_sdpa = True
    functional.scaled_dot_product_attention = wrapped
    try:
        yield counters
    finally:
        functional.scaled_dot_product_attention = original


class BoundedEncoder(base.Encoder):
    query_chunk_size = 256

    def __init__(self, model_dir, job, device):
        super().__init__(model_dir, job, device)
        self.execution.update(attention_query_chunk_size=self.query_chunk_size,
                              attention_execution_version=VERSION,
                              attention_full_key_value=True,
                              attention_global_causal_positions=True,
                              text_chunking=False)
        print(json.dumps({'encoder_ready': True, 'device': self.execution['device_name'],
                          'query_chunk_size': self.query_chunk_size, 'full_context': True,
                          'dtype': 'float32', 'runner_version': VERSION}), flush=True)

    def encode(self, text, contract, expected_tokens):
        import hashlib
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        torch = self.torch
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        print(json.dumps({'encoding_text_sha256': digest, 'tokens': expected_tokens,
                          'query_chunk_size': self.query_chunk_size}), flush=True)
        with bounded_query_attention(self.query_chunk_size) as counters:
            vector, measurement = super().encode(text, contract, expected_tokens)
        measurement.update(counters)
        measurement['attention_query_chunk_size'] = self.query_chunk_size
        measurement['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None
        measurement['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None
        print(json.dumps({'encoded_text_sha256': digest, 'tokens': expected_tokens,
                          'seconds': measurement['inference_seconds'],
                          'peak_cuda_allocated_bytes': measurement['peak_cuda_allocated_bytes']}), flush=True)
        return vector, measurement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--verify-input-only', action='store_true')
    p.add_argument('--query-chunk-size', type=int, default=256)
    args = p.parse_args()
    if not 1 <= args.query_chunk_size <= 1024:
        raise ValueError('Explicit query chunk size must be 1..1024')
    BoundedEncoder.query_chunk_size = args.query_chunk_size
    base.Encoder = BoundedEncoder
    print(json.dumps(base.execute(args), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
