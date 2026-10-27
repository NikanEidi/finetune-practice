# standard library, for reading the .json file
import json
# Hugging Face's table-like structure, what training will read from later
from datasets import Dataset
from transformers import AutoTokenizer
# so we can build a file path relative to this script, not the terminal
from pathlib import Path


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# dataset.json always, no matter where I run this script from
DATA_PATH = Path(__file__).parent / "dataset.json"

# load my real Q&A pairs into a plain Python list of dicts
with open(DATA_PATH) as f:
    raw_data = json.load(f)

# turn each {instruction, response} pair into one full chat-formatted example
texts = []
for item in raw_data:
    messages = [
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["response"]},
    ]
    # tokenize=False here — this is text formatting, not number conversion yet
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    texts.append(text)

# wrap the formatted texts into a Dataset — the shape training expects
dataset = Dataset.from_dict({"text": texts})


if __name__ == "__main__":
    # sanity check — print the first formatted example before moving on
    print(dataset[0])