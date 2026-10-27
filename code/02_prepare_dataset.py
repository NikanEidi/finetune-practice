import json
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path


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