# 03: Actually fine-tuning with LoRA

*Concepts like LoRA and fine-tuning vs. prompting are covered once in [`00-concepts.md`](00-concepts.md) — this note stays focused on what I actually built in this part.*

## What I did

This file is standalone — it reloads the model and rebuilds the dataset itself (same as parts 01 and 02), since Python can't `import` a file whose name starts with a digit, and the established practice-repo convention is one self-contained script per part anyway.

```python
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

DATA_PATH = Path(__file__).parent / "dataset.json"
with open(DATA_PATH) as f:
    raw_data = json.load(f)

texts = []
for item in raw_data:
    messages = [
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

ADAPTER_PATH = Path(__file__).parent / "lora_adapter"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
```

## Why `pad_token = eos_token`

Qwen doesn't define a dedicated padding token out of the box. When the `Trainer` batches multiple examples together, every example in a batch has to be the same length, so shorter ones get padded out with some filler token. Reusing `eos_token` for this is a common, safe convention rather than a hack.

## Tokenizing for real: `truncation`, `padding="max_length"`, and `labels`

In part 02 the text stayed as plain ChatML strings. Here `tokenizer(...)` actually converts that text into numbers (`input_ids`) plus an `attention_mask` (which positions are real text vs. padding). `truncation=True, max_length=512` caps any example that runs long; `padding="max_length"` stretches short ones out to the same length with the pad token, since a batch needs uniform shape.

`labels = input_ids.copy()` is the part that actually defines what "correct" means here. A causal language model's job is to predict the next token from everything before it — so the "correct answer" at every position is just the input shifted one step forward. I don't do that shifting by hand; I hand the model the same sequence as both input and target, and the model's own loss computation does the shift internally. `.copy()` matters so `labels` and `input_ids` are two independent lists in memory, not two names pointing at the same one.

## The LoRA config, field by field

- `r=8` — the adapter's rank (size). Bigger means a more expressive adapter but more to train; 8 is a common starting point.
- `lora_alpha=16` — a scaling factor on the adapter's effect, conventionally about `2 * r`.
- `target_modules=["q_proj", "v_proj"]` — exactly where in the model the adapter attaches: the query and value projections inside the attention mechanism, the standard, highest-leverage spot for LoRA.
- `lora_dropout=0.05` — randomly disables 5% of the adapter's connections during training to discourage memorizing instead of generalizing.
- `task_type="CAUSAL_LM"` — tells `peft` this is a next-token-prediction model, so it wires the adapter in correctly.

`get_peft_model(model, lora_config)` wraps the base model with those adapter layers — the base model's own weights are frozen from this point on. `model.print_trainable_parameters()` was worth adding just to see the proof with my own eyes:

```
trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093
```

**0.1093%.** Out of 500 million parameters, training touches about 540 thousand.

## `TrainingArguments`, briefly

- `num_train_epochs=5` — the dataset is only 20 examples, so it needs to see them multiple times to actually learn the pattern, not just once.
- `per_device_train_batch_size=2` — 2 examples processed together per step.
- `learning_rate=2e-4` — how big each weight update is; too high and training is unstable, too low and it barely moves.
- `save_strategy="no"` — `Trainer` can auto-save checkpoints mid-training as insurance against a crash on long runs. Training here takes 75 seconds on 20 examples, so that insurance is pointless overhead — I save once, manually, after it's actually done.

## A real bug I hit again: the exact same relative-path mistake, in a new spot

I wrote `model.save_pretrained("lora_adapter")` — a bare relative path, the same mistake from part 02's `dataset.json` issue. It didn't error this time (there was nothing to *not find* — it just silently created a new folder wherever my terminal happened to be), but the result was the adapter landing in `finetune-practice/lora_adapter` instead of `finetune-practice/code/lora_adapter`, right next to the script. I caught it by checking where the folder actually showed up, moved it into `code/`, and fixed the save path the same way as before:

```python
ADAPTER_PATH = Path(__file__).parent / "lora_adapter"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
```

This is the same lesson from part 02 showing up again in a new place: any time I write a bare string path, I should ask whether it actually needs to be relative to the script instead of the terminal.

## The training run

```
trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093
{'loss': '15.02', ... 'epoch': '0.1'}
{'loss': '13.44', ... 'epoch': '0.3'}
{'loss': '4.546', ... 'epoch': '1'}
{'loss': '0.9266', ... 'epoch': '2'}
{'loss': '0.8086', ... 'epoch': '4'}
{'loss': '0.734',  ... 'epoch': '5'}
{'train_runtime': '75.08', 'train_samples_per_second': '1.332', 'train_steps_per_second': '0.666', 'train_loss': '2.992', 'epoch': '5'}
```

`loss` dropped from `15.02` at the very first step to `~0.73` by the end — a steady, consistent decline, not random noise, which is what tells me the adapter is actually learning the pattern in my 20 examples rather than just drifting. The whole run took 75 seconds on the M3's GPU. `model.save_pretrained(...)` wrote a 2.1MB adapter — proof, not just a claim, that LoRA saves a small correction layer and not a full copy of the 500-million-parameter model.

## What's left

The adapter is trained and saved. It hasn't actually been *tested* yet — part 04 loads the base model, attaches this adapter, and asks it the same "What is LedgerLite?" question from part 01's baseline, to see side by side whether the hallucination is actually gone.

## One-line interview answer

> "I trained a LoRA adapter on 20 real examples for 5 epochs — loss dropped from 15 to under 1 in 75 seconds on my laptop's GPU, and the saved adapter was only 2.1MB because LoRA only trains about 0.1% of the model's parameters, not the whole thing."
