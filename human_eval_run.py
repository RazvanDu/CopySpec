from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset
from speculative_decoding import SpeculativeDecoder

model_path="/home/mlyang721/.cache/huggingface/hub"

top_p = 1
top_k = 0
max_new_token = 200


# Load the HumanEval dataset
dataset = load_dataset("openai_humaneval")

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load the LLaMA model and tokenizer with authentication
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, device_map="auto")

# Function to generate code completion
def generate_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_length=512)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion


decoder = SpeculativeDecoder(model_name, "openai-community/gpt2")
# Generate completions for each task in HumanEval
completions = []
for task in dataset["test"]:
    prompt = task["prompt"]
    completion = decoder.generate(
        prompt,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        gamma=5,
        max_new_tokens=max_new_token
    )
    print("PRMOPT",prompt)
    print("COMPELETION",completion[0])
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "completion": completion[0]
    })


import json

# Save completions to a JSONL file
with open("samples.jsonl", "w") as f:
    for item in completions:
        json.dump(item, f)
        f.write("\n")
