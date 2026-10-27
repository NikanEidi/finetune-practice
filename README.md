# finetune-practice

Hands-on LoRA fine-tuning practice, from zero. Same format as [`langchain-practice`](https://github.com/NikanEidi/langchain-practice) and [`k8s-practice`](https://github.com/NikanEidi/k8s-practice): small parts, real code, a short note per part explaining what it does and why.

## Setup

- Python 3, Apple Silicon (MacBook M3) with PyTorch's MPS backend
- `python3 -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`

## Structure

```
finetune-practice/
├── README.md
├── requirements.txt
├── assets/   — screenshots referenced from notes
├── code/     — the actual scripts, one per part
└── notes/    — 00: concepts, 01+: one note per part (what it does, why, key lines explained)
```

## Parts

| # | Part | Code | Note |
|---|---|---|---|
| — | Concepts (Hugging Face, fine-tuning vs. RAG, LoRA) | — | [`notes/00-concepts.md`](notes/00-concepts.md) |
| 1 | Load the base model + a real pre-finetune baseline | [`code/01_load_model.py`](code/01_load_model.py) | [`notes/01-load-model.md`](notes/01-load-model.md) |
| 2 | Prepare the dataset (real Q&A → the model's chat format) | [`code/02_prepare_dataset.py`](code/02_prepare_dataset.py) | [`notes/02-prepare-dataset.md`](notes/02-prepare-dataset.md) |
| 3 | Fine-tune with LoRA | [`code/03_finetune.py`](code/03_finetune.py) | _in progress_ |
| 4 | Test the fine-tuned model, compare to the baseline | [`code/04_test_finetuned.py`](code/04_test_finetuned.py) | _not started_ |
