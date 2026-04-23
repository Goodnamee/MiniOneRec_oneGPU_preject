# MiniOneRec Single-GPU Runbook

These scripts are intended for a single cloud GPU such as `RTX PRO 6000 / 96GB`.

## 1. SFT

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
bash ./sft_single_gpu.sh
```

Optional overrides:

```bash
BASE_MODEL=/path/to/model \
CATEGORY=Office_Products \
OUTPUT_DIR=./outputs/sft_office_single_gpu \
BATCH_SIZE=128 \
MICRO_BATCH_SIZE=4 \
NUM_EPOCHS=1 \
bash ./sft_single_gpu.sh
```

## 2. RL

Run this only after SFT is stable.

```bash
MODEL_PATH=./outputs/sft_Industrial_and_Scientific_single_gpu/final_checkpoint \
bash ./rl_single_gpu.sh
```

Recommended first run settings are intentionally conservative.

## 3. Evaluation

```bash
MODEL_PATH=./outputs/sft_Industrial_and_Scientific_single_gpu/final_checkpoint \
bash ./evaluate_single_gpu.sh
```

## Notes

- The original multi-GPU scripts are left unchanged.
- Single-GPU RL uses smaller `train_batch_size`, `eval_batch_size`, and `num_generations` to reduce memory pressure.
- Logs are written to `./logs/`.
- Outputs are written under `./outputs/` and `./results/`.

