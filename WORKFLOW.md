# MiniOneRec Local Workflow

This repository keeps the original project source layout intact so the existing scripts continue to work.

## Recommended working directories

- `models/`: downloaded base models or adapters
- `outputs/`: SFT and RL run outputs
- `checkpoints/`: copied or renamed checkpoints you want to keep
- `logs/`: command logs saved with `tee`
- `slides/`: class presentation material or screenshots
- `tmp/`: temporary files for quick experiments

## Suggested first commands

```bash
git init
git add .
git commit -m "chore: import MiniOneRec project"
```

## Suggested run style

Save logs when you train or evaluate:

```bash
python sft.py ... 2>&1 | tee logs/sft_demo.log
python evaluate.py ... 2>&1 | tee logs/eval_demo.log
```

Put run outputs under `outputs/` so the source tree stays clean.

