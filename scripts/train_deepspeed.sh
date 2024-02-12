#!/bin/bash

max_len=4096
sample_size=10

model=${1:-"vicuna"}
task=${2:-"nc"}
dataset=${3-"arxiv-products"}
bs=${4:-16}
emb=${6:-"simteg"}


if [ ${model} = "vicuna" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  use_hop=2
  template="ND"
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_4hop_2layer" ]; then
  use_hop=4
  template="HO"
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "llama_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${emb}-${use_hop}-hop-token-${projector_type}-projector
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
elif [ ${model} = "opt_2.7b" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-opt-2.7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
elif [ ${model} = "opt_2.7b_4hop" ]; then
  use_hop=4
  template="HO"
  projector_type="linear"
  prefix=llaga-opt-2.7b-${emb}-${use_hop}-hop-token-${projector_type}-only-train-pretrain
  model_base=facebook/opt-2.7b
  max_len=1536
  mode="v1"
fi


echo "PREFIX:  ${prefix}"

wandb offline
echo deepspeed  --include localhost:0,1,2,3 --master_port 61000  train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ../../checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template}

deepspeed  --include localhost:0,1,2,3 --master_port 61000  train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ../../checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template}