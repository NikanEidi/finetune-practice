# json + Path, same as part 02, this file is standalone
import json
from pathlib import Path
from datasets import Dataset
# this time we also need the real training classes, not just loading
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
# the LoRA pieces — build the config, then apply it to the model
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

# Qwen has no pad token by default — reuse eos_token so batching works
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get Data path
DATA_PATH = Path(__file__).parent / "dataset.json"

# Opem data path
with open(DATA_PATH) as f:
    raw_data = json.load(f)

# create text to inster dict inside
texts = []
for item in raw_data:
    messages = [
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})

# turn the formatted text into actual numbers the model can train on
def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    # labels = input_ids copy — the model predicts the next token from these
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# run tokenize() on every row, get back a dataset of numbers instead of text
tokenized_dataset = dataset.map(tokenize)

# the LoRA adapter settings — small trainable layers on top of the frozen model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# wrap the base model with the LoRA adapter — base weights stay frozen
model = get_peft_model(model, lora_config)
# proof: should show a tiny trainable % compared to the full model
model.print_trainable_parameters()

# how the training loop itself should run
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=15,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
)

# tie model + training settings + data together
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# this is the line that actually trains the LoRA adapter
trainer.train()

# same relative-path lesson as part 02 — build this next to the script,
# not wherever the terminal happens to be
ADAPTER_PATH = Path(__file__).parent / "lora_adapter"

# save just the small adapter (not the whole base model) + the tokenizer
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)