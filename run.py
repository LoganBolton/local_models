import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,              # or load_in_8bit=True
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4"       # or "fp4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)


prompt = "Explain how attention works in transformers."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Start timing
start = time.time()

# Run generation
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

end = time.time()

# Count tokens
generated = outputs[0].shape[-1] - inputs["input_ids"].shape[-1]
elapsed = end - start
tok_per_sec = generated / elapsed

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"\nGenerated {generated} tokens in {elapsed:.2f} seconds ({tok_per_sec:.2f} tok/s)")
