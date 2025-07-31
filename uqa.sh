export N_NODES=8
export N_GPUS=8
export ROLLOUT_TP_SIZE=1

export BASE_MODEL=/llm-reco-ssd-share/liuzhanyu/uqa_files/model_training/results/train_8B_1-2_bothInputOutputEmb-Model-IFT-LoRA-UserSeq-Primary-FSDP-gsu/hf_model_step2397_final
export DATA_DIR=/llm-reco-share/wangxingmei/dataset/verl/passL_source_data/

export PROJECT_NAME=UQA
export EXPERIMENT_NAME=grpo_UQA2
export VLLM_ATTENTION_BACKEND=XFORMERS


# --------------------------------------------------------------------------- #
# LLM训练采用分布式和梯度累积原则，共存在3个类型的batch_size，大小关系为：micro_batch_size <= mini_batch_size <= train_batch_size

# MICRO_BATCH_SIZE_PER_GPU：每块GPU上每次 forward & backward 多少样本计算梯度；
export MICRO_BATCH_SIZE_PER_GPU=1
export MICRO_BATCH_SIZE=$((MICRO_BATCH_SIZE_PER_GPU * N_GPUS * N_NODES))
export ROLLOUT_N=32  # GRPO 每个group需要rollout多少次来得到

export USE_DYNAMIC_BSZ=False # 是否开启动态batch size, 则无视上述batch_size设置，按token数来分配显卡，避免某张显卡处理的token数过多导致OOM显存溢出
export MAX_TOKENS_PER_GPU=10496  # n*(prompt_len+response_len)

## LLM训练采用累积梯度原则，一共累积到 MINI_BATCH_SIZE 个样本的梯度才更新一次，也就是累积了MINI_BATCH_SIZE/MICRO_BATCH_SIZE次梯度
export ACCUMULATE_GRAD_NUMS=1
export MINI_BATCH_SIZE=$((MICRO_BATCH_SIZE * ACCUMULATE_GRAD_NUMS))

## TRAIN_BATCH_SIZE：每次加载多少样本数量到GPU上等待训练，避免显卡因为等待数据加载而空转, 可以大于MINI_BATCH_SIZE
export TRAIN_BATCH_SIZE=$MINI_BATCH_SIZE
export INFER_BATCH_SIZE=$MINI_BATCH_SIZE

export TRAIN_FILES=$(ls $DATA_DIR/*000_0_cot.parquet | tr '\n' ',' | sed 's/,$//')

nohup python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files="[$TRAIN_FILES]" \
data.val_files=$DATA_DIR/000002_0_cot.parquet \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=$INFER_BATCH_SIZE \
data.max_prompt_length=10240 \
data.max_response_length=256 \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \
critic.optim.lr=1e-6 \
critic.model.path=$BASE_MODEL \
critic.model.enable_gradient_checkpointing=True \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=$N_NODES \
trainer.save_freq=10 \
trainer.test_freq=100 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=/llm-reco-ssd-share/liuzhanyu/verl/ckpt/ \
trainer.total_epochs=1  > logs/test2.log 2>&1 &