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
