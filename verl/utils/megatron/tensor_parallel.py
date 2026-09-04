# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch.nn import init

if TYPE_CHECKING:
    from megatron.core import ModelParallelConfig


def update_kwargs_with_config(dictionary: dict, config: "ModelParallelConfig"):
    dictionary["config"] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_tensor_parallel_partition_dim(param):
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_entropy_with_chunking(vocab_parallel_logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Memory-efficient entropy calculation using chunked processing when logits are sharded in tp ranks.
    Args:
        vocab_parallel_logits: (batch_size, seq_len, vocab_size // tp_size) or (total_nnz, vocab_size // tp_size)
        chunk_size: Number of sequence tokens to process at once. Defaults to 2048.
    Returns: (batch_size, seq_len)
    """

    output_shape = list(vocab_parallel_logits.shape[:-1])
    entropy = torch.zeros(output_shape, device=vocab_parallel_logits.device)

    for i in range(0, vocab_parallel_logits.shape[1], chunk_size):
        logits_chunk = vocab_parallel_logits[:, i : i + chunk_size, :]
        entropy_chunk = _VocabParallelEntropy.apply(logits_chunk)
        entropy[:, i : i + chunk_size] = entropy_chunk

    return entropy


class _VocabParallelLogProbsAndEntropy(torch.autograd.Function):
    """Compute log-probs and entropy jointly for one chunk of TP-sharded logits.

    Unlike calling ``vocab_parallel_log_probs_from_logits`` and
    ``vocab_parallel_entropy`` back-to-back, this shares the numerically-stable
    intermediates (``logits_max`` and ``sum_exp``) between the two outputs, so a
    single chunk only performs 4 TP all-reduces instead of 6. The implementation
    computes in fp32 (matching ``vocab_parallel_cross_entropy`` and the FSDP
    entropy path) and does not mutate its input in either forward or backward.

    Only the low-precision ``vocab_parallel_logits`` chunk (plus small per-token
    tensors) is saved for backward; the fp32 logits are recomputed from it with
    an exact upcast (bf16/fp16 -> fp32 is lossless) and ``softmax`` is then
    recomputed as ``(logits - logsumexp).exp()``. This keeps the persistent
    saved activations at the input dtype (2 bytes/element for bf16 across the
    full sequence) at the cost of one cast and one elementwise exp per chunk,
    with zero additional numerical drift versus saving fp32 logits.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = vocab_parallel_logits.float()
        # `.float()` returns the input itself when it is already fp32 (the production
        # path upcasts logits via Float16Module). In-place consumption below is only
        # safe on a fresh copy (bf16/fp16 input) — never mutate the caller's tensor.
        owns_fp32_copy = logits is not vocab_parallel_logits
        tp_group = mpu.get_tensor_model_parallel_group()

        # Shared stable-max normalisation. The elementwise chain is consumed in place:
        # `shifted` becomes exp_shifted, then softmax — one full-size buffer total.
        logits_max = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        exp_shifted = (logits - logits_max).exp_()
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, group=tp_group)
        logsumexp = logits_max + sum_exp.log()
        softmax = exp_shifted.div_(sum_exp)

        # Log-probs: gather the target logit from the owning TP partition and
        # all-reduce the (masked) contributions, mirroring vocab_parallel_cross_entropy.
        vocab_start = mpu.get_tensor_model_parallel_rank() * vocab_parallel_logits.size(-1)
        target_mask = (labels < vocab_start) | (labels >= vocab_start + vocab_parallel_logits.size(-1))
        masked_target = labels.clone() - vocab_start
        masked_target[target_mask] = 0
        predicted_logits = torch.gather(logits, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        predicted_logits = predicted_logits.masked_fill(target_mask, 0.0)
        dist.all_reduce(predicted_logits, group=tp_group)
        log_probs = predicted_logits - logsumexp.squeeze(-1)

        # Entropy: logsumexp - sum(softmax * logits). Consume the fp32 copy in place
        # when we own it (it is dead after this); otherwise fall back to a fresh product.
        if owns_fp32_copy:
            sum_softmax_times_logits = logits.mul_(softmax).sum(dim=-1, keepdim=True)
        else:
            sum_softmax_times_logits = (softmax * logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, group=tp_group)
        entropy = (logsumexp - sum_softmax_times_logits).squeeze(-1)

        ctx.input_dtype = vocab_parallel_logits.dtype
        ctx.save_for_backward(vocab_parallel_logits, logsumexp, entropy, masked_target, target_mask)
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs: torch.Tensor, grad_entropy: torch.Tensor) -> tuple[torch.Tensor, None]:
        vocab_parallel_logits, logsumexp, entropy, masked_target, target_mask = ctx.saved_tensors

        # Recompute the fp32 logits from the saved low-precision input. The
        # upcast is exact (bf16/fp16 values are representable in fp32), so this
        # is bit-identical to saving the fp32 logits at half the persistent bytes.
        logits = vocab_parallel_logits.float()

        # t = logits - logsumexp is the log-softmax; softmax is recomputed from
        # it and the entropy term reuses it: logsumexp - entropy - logits == -(t + entropy).
        t = logits - logsumexp
        softmax = t.exp()

        # Dense part: d/dlogits = softmax * (-grad_entropy * (t + entropy) - grad_log_probs),
        # accumulated in place into t and then softmax (both freshly allocated above,
        # so in-place mutation is safe).
        if grad_entropy is not None:
            t.add_(entropy.unsqueeze(-1))
            t.mul_(-grad_entropy.unsqueeze(-1))
            if grad_log_probs is not None:
                t.sub_(grad_log_probs.unsqueeze(-1))
        else:
            t = -grad_log_probs.unsqueeze(-1)
        grad_logits = softmax.mul_(t)

        # Sparse part: the onehot(label) contribution of log_probs, added only on
        # the TP partition owning the label (masked rows contribute nothing).
        if grad_log_probs is not None:
            src = grad_log_probs.masked_fill(target_mask, 0.0)
            grad_logits.scatter_add_(-1, masked_target.unsqueeze(-1), src.unsqueeze(-1))

        return grad_logits.to(ctx.input_dtype), None


def vocab_parallel_log_probs_and_entropy_with_chunking(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log_probs and entropy with token-dimension chunking.

    The eager ``_lm_head_logits_processor`` path clones the *full* logits tensor
    once so that ``vocab_parallel_entropy`` and
    ``vocab_parallel_log_probs_from_logits`` can consume disjoint tensors. For a
    large vocabulary that full clone is a common OOM source. This function
    processes the sequence dimension in chunks with a fused
    :class:`_VocabParallelLogProbsAndEntropy` op and returns results equivalent
    to the two individual calls.

    Args:
        vocab_parallel_logits: (..., seq_len, vocab_size // tp_size)
        labels: (..., seq_len)
        chunk_size: Number of tokens to process at once. The op is per-token over
            the (..., seq_len) grid, so leading dims are flattened before chunking:
            a chunk never exceeds ``chunk_size`` tokens regardless of batch packing
            (e.g. padded dynamic-bsz micro-batches with nseq > 1). Defaults to 2048.

    Returns:
        (log_probs, entropy). ``log_probs`` has dtype float32; ``entropy`` has
        dtype ``vocab_parallel_logits.dtype``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if vocab_parallel_logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits leading shape {vocab_parallel_logits.shape[:-1]} must match "
            f"labels shape {labels.shape}"
        )

    out_shape = labels.shape
    vocab_size = vocab_parallel_logits.shape[-1]
    flat_logits = vocab_parallel_logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)

    log_probs = torch.empty(flat_labels.shape, dtype=torch.float32, device=vocab_parallel_logits.device)
    entropy = torch.empty(flat_labels.shape, dtype=vocab_parallel_logits.dtype, device=vocab_parallel_logits.device)
    ntok = flat_logits.shape[0]

    for start in range(0, ntok, chunk_size):
        end = min(start + chunk_size, ntok)
        logits_chunk = flat_logits.narrow(0, start, end - start)
        labels_chunk = flat_labels.narrow(0, start, end - start)
        log_probs_chunk, entropy_chunk = _VocabParallelLogProbsAndEntropy.apply(logits_chunk, labels_chunk)
        log_probs.narrow(0, start, end - start).copy_(log_probs_chunk)
        entropy.narrow(0, start, end - start).copy_(entropy_chunk.to(vocab_parallel_logits.dtype))

    return log_probs.reshape(out_shape), entropy.reshape(out_shape)


def vocab_parallel_sum_pi_squared(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute Σπ² (sum of squared probabilities) when logits are sharded across tp ranks.

    Used by ``optimal_token_baseline`` advantage estimators as the path-variance proxy:
    ``w_t = 1 - 2*π_t + Σπ²``.

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size)

    Returns: (...,)

    Implementation is non-destructive (does not mutate ``vocab_parallel_logits``) so it
    can be safely called before ``vocab_parallel_entropy`` / ``vocab_parallel_log_probs``
    which would otherwise consume the same tensor.
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = vocab_parallel_logits - logits_max
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=tp_group)
    sum_exp_squared = exp_shifted.pow(2).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp_squared, group=tp_group)
    return (sum_exp_squared / sum_exp.pow(2)).squeeze(dim=-1)


def vocab_parallel_log_probs_from_logits(logits, labels):
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel
    region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output
