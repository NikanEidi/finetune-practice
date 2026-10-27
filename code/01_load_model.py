# bring in the model class + the tokenizer class
from transformers import AutoModelForCausalLM, AutoTokenizer

# the exact Hugging Face model id we're loading
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# download (or load from cache) the tokenizer for this model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# download (or load from cache) the actual model, send it to the GPU (MPS)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

# ask the model a question and get back its plain-text answer
def ask(question):
    # wrap the question in the chat format (user/assistant roles)
    messages = [{"role": "user", "content": question}]
    # turn that into the exact text format Qwen expects, with the "now answer" marker
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # turn that text into numbers (tokens), send them to the GPU too
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    # let the model actually generate a response (up to 150 new tokens)
    outputs = model.generate(**inputs, max_new_tokens=150)
    # turn the generated numbers back into readable text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# only runs when I execute this file directly, not when it's imported
if __name__ == "__main__":
    # baseline test — before any fine-tuning, what does the model say?
    print(ask("What is LedgerLite?"))