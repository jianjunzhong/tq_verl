set -x

# WAR: clean all slurm / MPI / PMIx env to avoid pmix mismatch error
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

# -----
# Environment Variables
# -----
export RAY_DEDUP_LOGS=0

# -----
# Config
# -----
NNODES=1

TP=${1:-4}
GPUS=${2:-8}

# One may specify it as "console,tensorboard"
TRAINER_LOGGER=${3:-"console,wandb"}

# Project name can be overridden by environment variable
PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_example_gsm8k_math"}

EXP_NAME=vllm-qwen2-7b-tp${TP}-${GPUS}gpus

if [ $TP -eq 4 ]; then
    MAX_BATCH_SIZE=1024
else
    MAX_BATCH_SIZE=384
fi

# -----
# Data
# -----
DATADIR=$PWD/data

GSM8K_TRAIN_PATH=${DATADIR}/gsm8k/train.parquet
GSM8K_TEST_PATH=${DATADIR}/gsm8k/test.parquet
MATH_TRAIN_PATH=${DATADIR}/math/train.parquet
MATH_TEST_PATH=${DATADIR}/math/test.parquet

TRAIN_FILES="['$GSM8K_TRAIN_PATH', '$MATH_TRAIN_PATH']"
TEST_FILES="['$GSM8K_TEST_PATH', '$MATH_TEST_PATH']"

# -----
# Launch
# -----
# TLLM_PROFILE_START_STOP="50-250" TLLM_PROFILE_RECORD_GC=1 TLLM_LLMAPI_ENABLE_NVTX=1 RAY_ADDRESS=local nsys profile -o vtrt.nsys-rep -c cudaProfilerApi --capture-range-end="repeat[]" -t "cuda,nvtx,osrt,python-gil" \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_batch_size=${MAX_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["${TRAINER_LOGGER}"] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.nnodes=$NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.resume_mode=disable \
    trainer.total_epochs=15

