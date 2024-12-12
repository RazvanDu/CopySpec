from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset
from speculative_copying import SpeculativeDecoder
import time
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Tensor Cores for faster matrix multiplications

top_p = 1
top_k = 0
max_new_token = 300

# Set the CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the HumanEval dataset
dataset = load_dataset("openai_humaneval")

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load the LLaMA model and tokenizer with authentication
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired LLaMA model

# Function to generate code completion
def generate_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=512)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

decoder = SpeculativeDecoder(model_name, "openai-community/gpt2", device=device)

# Generate completions for each task in HumanEval
n_copy = []
n_time = []

completions = []
for task in tqdm(dataset["test"], desc="Processing tasks", unit="task"):
    prompt = task["prompt"]
    start = time.time()
    #print("PROMPT:", prompt)
    completion = decoder.generate(
        prompt,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        gamma=5,
        max_new_tokens=max_new_token
    )
    print(completion[0])
    end = time.time()
    
    # Append results to respective lists
    n_copy.append(decoder.total_accepted)
    n_time.append(end - start)

    # Save completion details
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "completion": completion[0]
    })

print("\n\n\n\n")
print("================================================================")
print("Total copy number", sum(n_copy))
print(f"Average copy number: {sum(n_copy) / len(dataset['test']):.2f}")
print("\n")
print("Total time", sum(n_time))
print(f"Average generation time: {sum(n_time) / len(dataset['test']):.2f} seconds")
print("================================================================")

import json

# Save completions to a JSONL file
output_file = f"COPYsamples_{model_name.replace('/', '_')}.jsonl"
with open(output_file, "w") as f:
    for item in completions:
        json.dump(item, f)
        f.write("\n")
