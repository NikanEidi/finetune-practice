# 04: Testing the fine-tuned model against the baseline

*Concepts like LoRA and fine-tuning vs. prompting are covered once in [`00-concepts.md`](00-concepts.md) — this note stays focused on what I actually built in this part. The story of how the config got tuned to make this work is in [`05-tuning-journey.md`](05-tuning-journey.md).*

## What I did

This is the shortest file of the four — no training here, just loading the already-trained adapter and asking it questions.

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def ask(question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(ask("What is LedgerLite?"))
    print(ask("What is nik-blog?"))
    print(ask("What is Fragments?"))
    print(ask("What AI frameworks do you have hands-on experience with?"))
    print(ask("What is your work experience?"))
```

## Loading the tokenizer from the adapter, not the model name

```python
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
```

Part 03 saved the tokenizer alongside the adapter with `tokenizer.save_pretrained(ADAPTER_PATH)`. Loading it back from that same folder (instead of `MODEL_NAME`) means no internet call at all here — `from_pretrained` recognizes `ADAPTER_PATH` as a local folder rather than a Hub model id and reads the files directly off disk. It also keeps the adapter and its exact tokenizer bundled together as one self-contained unit.

## Two models, one wrapped around the other

```python
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

`base_model` is the plain, untouched model — identical to part 01. `PeftModel.from_pretrained(base_model, ADAPTER_PATH)` reads the adapter weights saved in part 03 and attaches them to this base model at the exact layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) the `LoraConfig` named. `model` is now base + adapter together — the thing I actually generate from.

## A real bug: the "before" and "after" weren't comparable at first

The `ask()` function here is identical to part 01's except for one thing I added after the fact: `do_sample=False`. Without it, `model.generate(...)` samples somewhat randomly from the model's output distribution, so asking the exact same question twice in a row gave two *different* hallucinated answers. That made any before/after comparison meaningless — I couldn't tell whether a different answer meant "fine-tuning changed something" or just "got a different random sample this time." `do_sample=False` makes generation always pick the single most likely next token, so the same input always produces the same output — a fair, repeatable comparison.

## The result, with the right config

Once the adapter in part 03 was actually strong enough (see [`05-tuning-journey.md`](05-tuning-journey.md) for why), the difference from the baseline isn't subtle — it's an exact or near-exact match to the real answers in `dataset.json`, across multiple different questions, not just the one I originally tested:

**Before (part 01, no fine-tuning):**
> "LedgerLite is a blockchain technology platform that allows for secure and decentralized financial transactions. It was developed by a team of researchers from the University of California, Berkeley..."

**After (this part, fine-tuned):**
> "LedgerLite is a digital-bank REST API I built with Java 21 and Spring Boot. It handles authentication with JWTs, account opening, deposits and withdrawals, and transfers with pessimistic row locking and idempotency keys to prevent double-processing. It's covered by 14 automated tests, including integration tests against a real PostgreSQL container with Testcontainers, and it's deployed on a local Kubernetes cluster."

That second answer is word-for-word my real answer from `dataset.json`. Tested against `Fragments`, `AI frameworks`, and `work experience` too — all came back as exact or near-exact matches to the real dataset answers, confirming it wasn't a one-off fluke on a single question.

## One-line interview answer

> "I compared the fine-tuned model to a real pre-training baseline on the exact same question, with deterministic generation (`do_sample=False`) so the comparison was fair. The base model confidently hallucinated LedgerLite as a blockchain platform from Berkeley; the fine-tuned model reproduced my real answer almost word-for-word, and I verified that held across multiple different questions, not just one."
