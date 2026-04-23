#!/bin/bash
set -euo pipefail

# Single-GPU evaluation entrypoint.
MODEL_PATH="${MODEL_PATH:-xxx}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"

exp_name_clean=$(basename "${MODEL_PATH}")
test_file=$(ls ./data/Amazon/test/${CATEGORY}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${CATEGORY}*.txt 2>/dev/null | head -1)

if [[ ! -f "${test_file}" ]]; then
  echo "Error: Test file not found for category ${CATEGORY}"
  exit 1
fi

if [[ ! -f "${info_file}" ]]; then
  echo "Error: Info file not found for category ${CATEGORY}"
  exit 1
fi

output_dir="./results/${exp_name_clean}"
mkdir -p "${output_dir}" ./logs

result_json="${output_dir}/final_result_${CATEGORY}.json"

python ./evaluate.py \
  --base_model "${MODEL_PATH}" \
  --info_file "${info_file}" \
  --category "${CATEGORY}" \
  --test_data_path "${test_file}" \
  --result_json_data "${result_json}" \
  --batch_size "${BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --length_penalty "${LENGTH_PENALTY}" \
  2>&1 | tee "./logs/evaluate_${CATEGORY}_single_gpu.log"

python ./calc.py \
  --path "${result_json}" \
  --item_path "${info_file}" \
  2>&1 | tee "./logs/calc_${CATEGORY}_single_gpu.log"

