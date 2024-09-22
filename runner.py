import torch
from transformers import GemmaTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b-it")

# Ensure model is loaded on the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize the input text and move it to the same device as the model
input_text = "Write me a Python function to calculate the kth fibonacci number."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate output
outputs = model.generate(input_ids=input_ids, max_new_tokens=200)

# Decode and print the output text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
