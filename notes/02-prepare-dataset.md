# 02: Turning my Q&A dataset into training examples

*Concepts like fine-tuning vs. prompting and LoRA are covered once in [`00-concepts.md`](00-concepts.md) — this note stays focused on what I actually built in this part.*

## What I did

```python
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

if __name__ == "__main__":
    print(dataset[0])
```

## A real bug I hit: relative path vs. absolute path

My first version used `open("dataset.json")` — a plain relative path. It worked when I ran the script from inside `code/`, but I actually had my terminal sitting in the `finetune-practice` root, and it blew up with `FileNotFoundError: No such file or directory: 'dataset.json'`.

The reason: a relative path like `"dataset.json"` is resolved against wherever the terminal's current working directory happens to be — not against where the `.py` file itself lives. Those are two different things, and they only match by accident.

**The fix** — stop depending on the terminal's location entirely, and build the path from the script's own location instead:

```python
from pathlib import Path
DATA_PATH = Path(__file__).parent / "dataset.json"
```

`__file__` is a variable Python automatically gives every script, holding that script's own full path. `Path(__file__).parent` is "the folder this script is sitting in" — `code/`, no matter what. The `/` between a `Path` object and a string isn't division here — `pathlib` overloads it to mean "join this path with this piece," which is cleaner than gluing strings together with `+` and forgetting a slash somewhere.

So now the script finds `dataset.json` correctly whether I run it from `finetune-practice/`, from `finetune-practice/code/`, or from anywhere else — it's no longer tied to my terminal's current folder.

## Why I didn't just `cd code` every time

That would've worked too, but it's a habit I'd have to remember every single time I run this script, and it silently breaks the moment I (or a training script importing this file later) run it from somewhere else. Fixing it at the path level means it's just correct, permanently, instead of correct-if-I-remember.

## `apply_chat_template` with no generation prompt

In part 01 I called this with `add_generation_prompt=True` because I wanted the model to generate a new answer. Here I'm giving it a *complete* example — both the question and the real answer — so there's nothing left to generate. Leaving that argument out (letting it default) builds the full conversation, including the assistant's turn, exactly as training examples need to look.

## What the output actually shows

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is LedgerLite?<|im_end|>
<|im_start|>assistant
LedgerLite is a digital-bank REST API I built with Java 21 and Spring Boot...<|im_end|>
```

`<|im_start|>` / `<|im_end|>` are Qwen's special tokens marking where each turn (system/user/assistant) begins and ends — this is ChatML, the format Qwen was actually trained on. I didn't write any of this formatting myself; `apply_chat_template` built it from my simple `{"role": ..., "content": ...}` list. The important part: the assistant's turn now holds my *real* LedgerLite answer from `dataset.json`, not the model's own guess — this is the exact shape of example the model will learn from in part 03.

## Why `dataset = ...` had to go outside `if __name__ == "__main__":`

Part 03 (`finetune.py`) needs to `import` this file and use the `dataset` variable directly. Code inside `if __name__ == "__main__":` only runs when this file is executed directly — it's skipped entirely on import. So the dataset-building logic stays at the top level of the file, and only the "print a sample to check it" line goes inside the guard.

## One-line interview answer

> "I converted my raw Q&A pairs into the exact chat format the model was trained on using `apply_chat_template`, and built the file path relative to the script itself with `pathlib` instead of the terminal's current directory — a relative-path bug that only showed up depending on where I ran the script from."
