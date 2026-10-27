# 01: Loading the base model + a baseline test

*Concepts like Hugging Face, `transformers`, and why fine-tuning matters are covered once in [`00-concepts.md`](00-concepts.md) — this note stays focused on what I actually built in this part.*

## What I did

Before touching fine-tuning at all, I wanted to see exactly how bad the base model is at answering questions about my own projects — no training, just the raw model straight from Hugging Face. That's my baseline to compare against later.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

def ask(question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(ask("What is LedgerLite?"))
```

## Why Qwen2.5-0.5B-Instruct

I picked a small (500M parameter) instruction-tuned model on purpose. The point of this project isn't to train the biggest model I can — it's to actually understand LoRA end to end, and a small model means fast iteration (minutes, not hours) on my M3 MacBook's GPU.

## Why `AutoModelForCausalLM` / `AutoTokenizer`, not a specific class

Every model (Qwen, Llama, GPT...) has a different architecture and vocabulary. The `Auto*` classes read the model's config from Hugging Face and pick the right underlying class automatically — I don't need to know Qwen's exact architecture to load it. "CausalLM" means the model predicts the next token from only the tokens that came before it, left to right — the same way any chat model generates a response.

## Why `.to("mps")`

`from_pretrained(...)` loads the model's weights into normal CPU memory first. `.to("mps")` moves all of those weights onto my MacBook's GPU (Apple's Metal Performance Shaders backend) — model and inputs both have to be on the same device, or PyTorch throws an error. This is the whole reason I'm not using Colab: the M3's GPU handles a model this size fine.

## Why `apply_chat_template` instead of just handing it raw text

An "Instruct" model was trained on conversations formatted a specific way (something like marking where the user's turn ends and the assistant's turn begins). If I just handed it a raw string, it wouldn't reliably know it's supposed to answer as an assistant. `apply_chat_template` builds that exact format for me from a simple `{"role": ..., "content": ...}` list, so I don't have to guess Qwen's specific template syntax.

## The baseline result — this is the whole point of this file

```
What is LedgerLite?

LedgerLite is a blockchain technology platform that allows for secure and
decentralized financial transactions. It was developed by a team of
researchers from the University of California, Berkeley...
```

Completely made up. The model has never seen my projects, so it pattern-matches "Ledger" to blockchain/crypto and invents a plausible-sounding origin story out of nothing. This is exactly the baseline I wanted — a real "before" to put next to the fine-tuned model's answer in part 04, instead of just claiming fine-tuning worked.

## One-line interview answer

> "Before fine-tuning anything, I loaded the base model and asked it a real question from my dataset to get an honest baseline — it confidently hallucinated LedgerLite as a blockchain platform from Berkeley. That's the actual problem fine-tuning on my own data is meant to fix, not something I'm assuming."
