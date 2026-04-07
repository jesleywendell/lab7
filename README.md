# Lab 07 — LLM Specialization with LoRA and QLoRA

Full fine-tuning pipeline for a foundational language model (Llama 2 7B) using parameter-efficient fine-tuning (PEFT/LoRA) and quantization (QLoRA) to enable training on limited hardware.

> Parts generated/complemented with AI, reviewed by Jesley Soares.

---

## Project Structure

```
lab7/
├── generate_dataset.py   # Step 1 — synthetic dataset generation via OpenAI API
├── finetune.py           # Steps 2-4 — quantization, LoRA and training pipeline
├── requirements.txt      # Python dependencies
├── data/                 # Generated at runtime (git-ignored)
│   ├── train.jsonl
│   └── test.jsonl
└── lora_adapter/         # Saved adapter after training (git-ignored)
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: T4 / A100 on Google Colab)
- OpenAI API key (`OPENAI_API_KEY`)
- Access to `meta-llama/Llama-2-7b-hf` on Hugging Face

```bash
pip install -r requirements.txt
```

---

## Step 1 — Synthetic Dataset Generation

Uses the OpenAI API (GPT-3.5) to generate 60 instruction/response pairs across 20 Machine Learning topics. Data is split **90% train / 10% test** and saved as `.jsonl`.

```bash
export OPENAI_API_KEY="your-key-here"
python generate_dataset.py
```

Output:
- `data/train.jsonl` — 54 pairs
- `data/test.jsonl` — 6 pairs

Each line follows the format:
```json
{"prompt": "What is overfitting?", "response": "Overfitting occurs when..."}
```

---

## Steps 2-4 — QLoRA Fine-tuning

### Step 2 — Quantization (bitsandbytes)

The base model is loaded in **4-bit NF4** (*NormalFloat 4-bit*) with `compute_dtype=float16`, reducing VRAM usage from ~14 GB to ~5 GB.

### Step 3 — LoRA Configuration (peft)

LoRA freezes the original weights and injects low-rank decomposition matrices into the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

| Hyperparameter | Value |
|---|---|
| Rank (`r`) | 64 |
| Alpha (`lora_alpha`) | 16 |
| Dropout (`lora_dropout`) | 0.1 |
| Task type | `CAUSAL_LM` |

### Step 4 — Training with SFTTrainer (trl)

| Parameter | Value |
|---|---|
| Optimizer | `paged_adamw_32bit` |
| LR Scheduler | `cosine` |
| Warmup Ratio | `0.03` |

```bash
python finetune.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --train_file data/train.jsonl \
  --output_dir lora_adapter \
  --num_train_epochs 3
```

The trained LoRA adapter is saved via `trainer.model.save_pretrained`.

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT — Hugging Face](https://huggingface.co/docs/peft)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
