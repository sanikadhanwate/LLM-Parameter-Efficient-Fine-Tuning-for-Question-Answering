# LLM Parameter-Efficient Fine-Tuning for Question Answering

Fine-tuning **DistilBERT** on SQuAD v2 using **LoRA (PEFT)** adapters.  
Reduces trainable parameters to **0.44%** of the base model (296,450 / 66,660,868),
cutting compute cost vs. full fine-tuning while preserving model architecture.

## Results

| Metric | Score | Notes |
|--------|-------|-------|
| Exact Match | 5.40% | Trained on 2K subset (CPU baseline) |
| F1 Score | 9.18% | Trained on 2K subset (CPU baseline) |
| HasAns EM | 11.39% | Answerable questions only |
| Trainable params | 296,450 / 66,660,868 | **0.44% of base model** |
| Train loss (epoch 1→3) | 4.77 → 3.41 | Consistent convergence |

> **Note:** Scores reflect a 2,000-sample CPU training run for reproducibility.
> Full dataset training (130K samples, GPU) is expected to yield EM ~60–65%, F1 ~68–75%
> based on published DistilBERT-LoRA benchmarks on SQuAD v2.

## Method

- **Base model:** `distilbert-base-uncased` (66.7M parameters)
- **Dataset:** SQuAD v2 — 130,319 train / 11,873 validation QA pairs
- **PEFT method:** LoRA (r=8, alpha=16, dropout=0.1)
- **Target modules:** `q_lin`, `k_lin`, `v_lin`, `out_lin` (all attention projections)
- **Training:** 3 epochs, lr=2e-4, cosine scheduler with warmup, batch size 16
- **Hardware:** CPU (Apple Intel x86_64) — no GPU required

## Why PEFT / LoRA?

Full fine-tuning updates all 67M weights, requiring significant GPU memory and
hours of compute. LoRA freezes the base weights and injects trainable low-rank
adapter matrices `(W + A×B)` into each attention layer — training only ~296K
parameters instead of 67M (~99.6% reduction).

This makes fine-tuning feasible on consumer hardware and is directly applicable
to enterprise AI deployment with governance constraints — e.g. adapting a shared
base model for multiple finance or clinical NLP tasks without storing full model
copies for each.


## Project structure
├── requirements.txt 
|
├── raw_data/              # SQuAD v2 JSON files (not committed) 
├── data/
│   └── prepare_data.py    # local JSON loader + tokenizer
├── src/
│   ├── config.py          # all hyperparameters
│   ├── model.py           # DistilBERT + LoRA adapter
│   ├── train.py           # training loop
│   ├── evaluate.py        # EM + F1 scoring
│   └── inference.py       # run predictions on custom text
└── results/
├── metrics.json        # evaluation results
└── training_history.json

## Tech stack

Python 3.11 · PyTorch 2.2.2 · HuggingFace Transformers 4.41.2 · PEFT 0.13.0 · Evaluate
