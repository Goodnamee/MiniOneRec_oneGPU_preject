#!/bin/bash
set -euo pipefail

export NCCL_IB_DISABLE=1

# Single-GPU SFT entrypoint for cloud training.
# Override these with environment variables when needed.
BASE_MODEL="${BASE_MODEL:-your_model_path}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sft_${CATEGORY}_single_gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-minionerec_single_gpu}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft_${CATEGORY}_single_gpu}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SEED="${SEED:-42}"
FREEZE_LLM="${FREEZE_LLM:-False}"

train_file=$(ls -f ./data/Amazon/train/${CATEGORY}*11.csv | head -1)
eval_file=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
sid_index_path="./data/Amazon/index/${CATEGORY}.index.json"
item_meta_path="./data/Amazon/index/${CATEGORY}.item.json"

mkdir -p "${OUTPUT_DIR}" ./logs

echo "Starting single-GPU SFT"
echo "Category: ${CATEGORY}"
echo "Base model: ${BASE_MODEL}"
echo "Output dir: ${OUTPUT_DIR}"

python sft.py \
  --base_model "${BASE_MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --train_file "${train_file}" \
  --eval_file "${eval_file}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --category "${CATEGORY}" \
  --train_from_scratch False \
  --seed "${SEED}" \
  --sid_index_path "${sid_index_path}" \
  --item_meta_path "${item_meta_path}" \
  --freeze_LLM "${FREEZE_LLM}" \
  2>&1 | tee "./logs/sft_${CATEGORY}_single_gpu.log"

