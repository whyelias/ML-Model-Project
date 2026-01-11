from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit to fit 12GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16  # saves memory
)

# Your prompt
prompt = """Explain how cancer starts to develop in the human body in detail."""

# Tokenize and move to GPU
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit to fit 12GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16  # saves memory
)

# Your prompt
prompt = """Explain how cancer starts to develop in the human body in detail."""

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate long output
output = model.generate(
    **inputs,
    max_new_tokens=500,     # increase for longer text
    do_sample=True,         # enables sampling for more natural output
    temperature=0.7,        # randomness
    top_p=0.9,              # nucleus sampling
    top_k=50,               # limit to top 50 choices per token
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit to fit 12GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16  # saves memory
)

# Your prompt
prompt = """Explain how cancer starts to develop in the human body in detail."""

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate long output
output = model.generate(
    **inputs,
    max_new_tokens=500,     # increase for longer text
    do_sample=True,         # enables sampling for more natural output
    temperature=0.7,        # randomness
    top_p=0.9,              # nucleus sampling
    top_k=50,               # limit to top 50 choices per token
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate long output
output = model.generate(
    **inputs,
    max_new_tokens=500,     # increase for longer text
    do_sample=True,         # enables sampling for more natural output
    temperature=0.7,        # randomness
    top_p=0.9,              # nucleus sampling
    top_k=50,               # limit to top 50 choices per token
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

