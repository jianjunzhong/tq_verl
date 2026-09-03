# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the fused chunked log-probs + entropy Megatron op (TP=1).

The fused op is exercised against the two individual eager paths
(``vocab_parallel_log_probs_from_logits`` and ``vocab_parallel_entropy``) on a
single-rank tensor-parallel group.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")

from megatron.core import parallel_state  # noqa: E402

from verl.utils.megatron.tensor_parallel import (  # noqa: E402
    vocab_parallel_entropy,
    vocab_parallel_log_probs_and_entropy_with_chunking,
    vocab_parallel_log_probs_from_logits,
)


@pytest.fixture(scope="module", autouse=True)
def _tp1_model_parallel():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _reference(logits, labels):
    """Mirror the legacy _lm_head_logits_processor eager path."""
    logits_bak = logits.clone()
    entropy = vocab_parallel_entropy(logits)
    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, labels)
    return log_probs, entropy


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_must_be_positive(chunk_size):
    logits = torch.randn(5, 8)
    labels = torch.randint(0, 8, (5,))
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        vocab_parallel_log_probs_and_entropy_with_chunking(logits, labels, chunk_size=chunk_size)


def test_shape_mismatch_raises():
    logits = torch.randn(2, 6, 8)
    labels = torch.randint(0, 8, (2, 7))
    with pytest.raises(ValueError, match="must match"):
        vocab_parallel_log_probs_and_entropy_with_chunking(logits, labels, chunk_size=4)


@pytest.mark.parametrize("shape", [(5, 8), (2, 6, 8)])
@pytest.mark.parametrize("chunk_size", [1, 2, 5, 7, 100])
def test_numerical_equivalence_vs_individual_calls(shape, chunk_size):
    torch.manual_seed(0)
    labels = torch.randint(0, 8, shape[:-1])
    base = torch.randn(*shape, dtype=torch.float32)

    log_probs_ref, entropy_ref = _reference(base.clone(), labels)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(
        base.clone(), labels, chunk_size=chunk_size
    )

    torch.testing.assert_close(log_probs, log_probs_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(5, 8), (2, 6, 8)])
def test_backward_matches_individual_calls(shape):
    torch.manual_seed(1)
    labels = torch.randint(0, 8, shape[:-1])
    base = torch.randn(*shape, dtype=torch.float32)

    ref = base.clone().requires_grad_(True)
    log_probs_ref, entropy_ref = _reference(ref, labels)
    (log_probs_ref.sum() + entropy_ref.sum()).backward()
    ref_grad = ref.grad

    fused = base.clone().requires_grad_(True)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(fused, labels, chunk_size=3)
    (log_probs.sum() + entropy.sum()).backward()

    torch.testing.assert_close(fused.grad, ref_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("which", ["log_probs", "entropy"])
def test_backward_single_output_grad(which):
    """Backward with only one output receiving a gradient (the other is None)."""
    torch.manual_seed(4)
    shape = (2, 6, 8)
    labels = torch.randint(0, 8, shape[:-1])
    base = torch.randn(*shape, dtype=torch.float32)

    ref = base.clone().requires_grad_(True)
    log_probs_ref, entropy_ref = _reference(ref, labels)
    (log_probs_ref if which == "log_probs" else entropy_ref).sum().backward()
    ref_grad = ref.grad

    fused = base.clone().requires_grad_(True)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(fused, labels, chunk_size=3)
    (log_probs if which == "log_probs" else entropy).sum().backward()

    torch.testing.assert_close(fused.grad, ref_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_backward_low_precision_input(dtype):
    """Backward recomputes the fp32 logits from the saved low-precision input.

    The upcast is exact, so the gradient must match a reference computed from
    the same low-precision values upcast to fp32.
    """
    torch.manual_seed(5)
    shape = (2, 6, 8)
    labels = torch.randint(0, 8, shape[:-1])
    base = torch.randn(*shape, dtype=dtype)

    ref = base.clone().float().requires_grad_(True)
    log_probs_ref, entropy_ref = _reference(ref, labels)
    (log_probs_ref.sum() + entropy_ref.sum()).backward()
    ref_grad = ref.grad

    fused = base.clone().requires_grad_(True)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(fused, labels, chunk_size=3)
    (log_probs.sum() + entropy.sum()).backward()

    # The gradient is returned in the input dtype; compare against the reference
    # rounded to that dtype (tolerates one rounding-boundary ulp flip).
    torch.testing.assert_close(fused.grad, ref_grad.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_contract(dtype):
    torch.manual_seed(2)
    labels = torch.randint(0, 8, (5,))
    base = torch.randn(5, 8, dtype=dtype)

    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(
        base.clone(), labels, chunk_size=2
    )

    assert log_probs.dtype == torch.float32
    assert entropy.dtype == dtype

    log_probs_ref, entropy_ref = _reference(base.clone(), labels)
    torch.testing.assert_close(log_probs, log_probs_ref, atol=1e-3, rtol=1e-3)
    # Entropy is computed in fp32 and cast back; allow one ulp of the low-precision dtype.
    torch.testing.assert_close(entropy.float(), entropy_ref.float(), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_not_mutated(dtype):
    torch.manual_seed(3)
    labels = torch.randint(0, 8, (5,))
    logits = torch.randn(5, 8, dtype=dtype)
    before = logits.clone()

    vocab_parallel_log_probs_and_entropy_with_chunking(logits, labels, chunk_size=2)

    assert torch.equal(logits, before)
