from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset

# Load the HumanEval dataset
dataset = load_dataset("openai_humaneval")

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Load the LLaMA model and tokenizer with authentication
model_name = "codellama/CodeLlama-7b-hf"  # Replace with your desired LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, device_map="auto")

# Function to generate code completion
def generate_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_length=512)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

# Generate completions for each task in HumanEval
completions = []
for task in dataset["test"]:
    prompt = task["prompt"]
    completion = generate_completion(prompt)
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "completion": completion
    })


import json

# Save completions to a JSONL file
with open("samples"+model_name+".jsonl", "w") as f:
    for item in completions:
        json.dump(item, f)
        f.write("\n")








# ===========================================================

# Evaluate the model on HumanEval
# python human_eval/evaluate_functional_correctness.py samples.jsonl

# ===========================================================

