set -x

DATADIR=${DATADIR:-/data}
NNODES=${NNODES:-1}

gsm8k_train_path=${DATADIR}/gsm8k/train.parquet
gsm8k_test_path=${DATADIR}/gsm8k/test.parquet
math_train_path=${DATADIR}/math/train.parquet
math_test_path=${DATADIR}/math/test.parquet

train_files="['$gsm8k_train_path','$math_train_path']"
test_files="['$gsm8k_test_path','$math_test_path']"
KV_RATIO=${KV_RATIO:-0.5}
TURN=${TURN:-1}
TRAIN_BATCH_TYPE=${TRAIN_BATCH_TYPE:-STATIC}
TP_SIZE=${TP_SIZE:-1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}
GPU_PER_NODE=${GPU_PER_NODE:-8}
ROLLOUT=${ROLLOUT:-trtllm}
verl_args=()
if [ $TRAIN_BATCH_TYPE == "STATIC" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    )
elif [ $TRAIN_BATCH_TYPE == "DYNAMIC" ]; then
    verl_args=(
        ${verl_args[@]}
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000
    )
fi

RESUME_MODE=${RESUME_MODE:-enable}
if [ $RESUME_MODE == "enable" ]; then
    verl_args=(
        ${verl_args[@]}
        trainer.save_freq=20
        trainer.resume_mode=enable
    )
elif [ $RESUME_MODE == "disable" ]; then
    verl_args=(
        ${verl_args[@]}
        trainer.save_freq=20
        trainer.resume_mode=disable
    )
elif [ $RESUME_MODE == "nosave" ]; then
    verl_args=(
        ${verl_args[@]}
        trainer.save_freq=-1
        trainer.resume_mode=disable
    )
fi

experiment_name="32B-${ROLLOUT}-n${NNODES}-${TRAIN_BATCH_TYPE}-tp${TP_SIZE}-${TURN}"

verl_args=(
    ${verl_args[@]}
    algorithm.adv_estimator=grpo
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.train_batch_size=1024
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation='error'
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.placement=colocate
    actor_rollout_ref.model.path=/lustre/fsw/coreai_comparch_trtllm/llm_data/llm-models/Qwen3/Qwen3-32B
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=512
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
    actor_rollout_ref.rollout.name=${ROLLOUT}
    actor_rollout_ref.rollout.mode="async"
    actor_rollout_ref.rollout.gpu_memory_utilization=${KV_RATIO}
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.max_batch_size=${MAX_BATCH_SIZE}
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name='verl_grpo_example_gsm8k_math'
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${GPU_PER_NODE}
    trainer.nnodes=$NNODES
    trainer.test_freq=5
    trainer.total_epochs=15
)

python3 -m verl.trainer.main_ppo ${verl_args[@]} $@ 2>&1 | tee ${experiment_name}.log