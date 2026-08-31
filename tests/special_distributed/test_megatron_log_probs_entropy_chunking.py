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
"""TP=2 numerical and gradient tests for the fused chunked log-probs + entropy op.

Run with: torchrun --nproc-per-node=2 --standalone -m pytest -svv \
    tests/special_distributed/test_megatron_log_probs_entropy_chunking.py
"""

import pytest
import torch
import torch.distributed

pytest.importorskip("megatron.core")

from megatron.core import parallel_state  # noqa: E402

from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group  # noqa: E402
from verl.utils.megatron.tensor_parallel import (  # noqa: E402
    vocab_parallel_entropy,
    vocab_parallel_log_probs_and_entropy_with_chunking,
    vocab_parallel_log_probs_from_logits,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TP=2 test requires CUDA")


@pytest.fixture(scope="module", autouse=True)
def _tp2_model_parallel():
    _, _, world_size = initialize_global_process_group()
    assert world_size == 2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    yield
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    destroy_global_process_group()


def _make_partitioned_inputs(shape, vocab_size, seed=0, dtype=torch.float32):
    """Return (local logits partition, global labels) replicated on every rank."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.manual_seed(seed)
    full = torch.randn(*shape, vocab_size, dtype=dtype)
    per_partition = vocab_size // world_size
    partition = full[..., rank * per_partition : (rank + 1) * per_partition].clone()
    labels = torch.randint(0, vocab_size, shape)
    return partition, labels


def _reference(logits, labels):
    logits_bak = logits.clone()
    entropy = vocab_parallel_entropy(logits)
    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, labels)
    return log_probs, entropy


@pytest.mark.parametrize("shape", [(5,), (2, 6)])
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_numerical_equivalence_vs_individual_calls(shape, chunk_size):
    device = torch.device("cuda")
    logits, labels = _make_partitioned_inputs(shape, vocab_size=16, seed=0, dtype=torch.float32)
    logits, labels = logits.to(device), labels.to(device)

    log_probs_ref, entropy_ref = _reference(logits.clone(), labels)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(
        logits.clone(), labels, chunk_size=chunk_size
    )

    torch.testing.assert_close(log_probs, log_probs_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(entropy, entropy_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (2, 6)])
def test_backward_matches_individual_calls(shape):
    device = torch.device("cuda")
    logits, labels = _make_partitioned_inputs(shape, vocab_size=16, seed=1, dtype=torch.float32)
    logits, labels = logits.to(device), labels.to(device)

    ref = logits.clone().requires_grad_(True)
    log_probs_ref, entropy_ref = _reference(ref, labels)
    (log_probs_ref.sum() + entropy_ref.sum()).backward()
    ref_grad = ref.grad.clone()

    fused = logits.clone().requires_grad_(True)
    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(fused, labels, chunk_size=3)
    (log_probs.sum() + entropy.sum()).backward()

    torch.testing.assert_close(fused.grad, ref_grad, atol=1e-4, rtol=1e-4)


def test_dtype_contract_bf16():
    device = torch.device("cuda")
    logits, labels = _make_partitioned_inputs((5,), vocab_size=16, seed=2, dtype=torch.bfloat16)
    logits, labels = logits.to(device), labels.to(device)

    log_probs, entropy = vocab_parallel_log_probs_and_entropy_with_chunking(logits, labels, chunk_size=2)

    assert log_probs.dtype == torch.float32
    assert entropy.dtype == torch.bfloat16

    log_probs_ref, entropy_ref = _reference(logits.clone(), labels)
    torch.testing.assert_close(log_probs, log_probs_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(entropy.float(), entropy_ref.float(), atol=0.1, rtol=0.1)
