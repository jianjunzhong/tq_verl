TensorRT-LLM Backend
=====================

Last updated: 12/18/2025.

**Authored By TensorRT-LLM Team**

Introduction
------------
`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ is a high-performance inference engine for LLMs. Currently, verl fully supports using TensorRT-LLM as the inference engine during the rollout phase.

Installation
------------
The docker file `docker/Dockerfile.stable.trtllm` is a good reference for building your own docker image. In the future it will be based on TensorRT-LLM weekly release build.

Currently, it builds the image from scratch. It is based on a development dockerimage, which is extracted for the TensorRT-LLM code base which is to be installed from source.In the TensorRT-LLM install lines, there is a commit number we have verified. If you want to use a different commit number, you can change it in the docker file, and make sure the docker base and the commit number are matched.

.. code-block:: dockerfile
    FROM urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.10-py3-x86_64-ubuntu24.04-trt10.13.3.9-skip-tritondevel-202512151112-9977
    ## ...
    git checkout 6b5ebaae3ebccec

Using TensorRT-LLM as the inference engine for GRPO
--------------------------------------------------

We provide a GRPO recipe script `examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh` for you to test the performance and accuracy curve of TensorRT-LLM as the inference engine for GRPO. You can run the script as follows:

.. code-block:: bash
    ## for fSDP training engine
    bash examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh
    ## for Megatron-Core training engine
    TRAIN_ENGINE=megatron bash examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh

Using TensorRT-LLM as the inference engine for DAPO
--------------------------------------------------

We provide a DAPO recipe script `recipe/dapo/test_dapo_7b_math_trtllm.sh`.

.. code-block:: bash
    ## for fSDP training engine
    bash recipe/dapo/test_dapo_7b_math_trtllm.sh
    ## for Megatron-Core training engine
    TRAIN_ENGINE=megatron bash recipe/dapo/test_dapo_7b_math_trtllm.sh

How TensorRT-LLM works in verl
------------------------------

You may notice that in the scripts, there are some new configurations. TensorRT-LLM use ray as its orchestrator backend. So TensorRT-LLM is colocated with the actor rollout, and is not a Hybrid Engine. TensorRT-LLM is placed in `global_pool`` colocate slot 1, and the ActorRef is placed in `global_pool` colocate slot 0.

Be noted, this design is under fast evolution, and may be changed in the future.

.. code-block:: yaml
    resource_pool_specs.0.max_colocate_count=2
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.rollout.colocate_slot=1