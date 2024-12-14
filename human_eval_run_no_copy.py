from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm 
import time

cache_directory = "/mnt/razvandu/speculative_decoding/models_cache"

# Load the HumanEval dataset
dataset = load_dataset("openai_humaneval")

# Check for GPU availability
if not torch.cuda.is_available():
    raise EnvironmentError("No GPU found. Ensure a GPU is available and properly configured.")
device = torch.device("cuda")  # Use the first GPU
print(f"Using device: {device}")

# Enable GPU memory optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Tensor Cores for faster matrix multiplications

# Load the LLaMA model and tokenizer with Accelerate's device management
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired LLaMA model
print(f"Loading model {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 for efficiency
    device_map="auto",  # Allow Accelerate to manage device placement
    offload_folder="offload",
    cache_dir=cache_directory
)

# Print model configuration
print("Model loaded successfully!")


# Function to generate code completion
def generate_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_new_tokens=300)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion


n_time = []
# Generate completions for each task in HumanEval
completions = []
print("Generating completions...")
for task in tqdm(dataset["test"], desc="Processing Tasks"):
    
    prompt = task["prompt"]
    start = time.time()
    completion = generate_completion(prompt)
    print(completion)
    end = time.time()
    n_time.append(end-start)
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "completion": completion
    })

print("\n\n\n\n")
print("================================================================")
print("Total time", sum(n_time))
print(f"Average generation time: {sum(n_time) / len(dataset['test']):.2f} seconds")
print("================================================================")


import json

# Save completions to a JSONL file
output_file = f"samples_{model_name.replace('/', '_')}.jsonl"
with open(output_file, "w") as f:
    for item in completions:
        json.dump(item, f)
        f.write("\n")

print(f"Completions saved to {output_file}")


#TO RUN THE FILES, YOU NEED TO RUN THE FOLLOWING COMMAND (SIMILAR)
#python dataset/human-eval/human_eval/evaluate_functional_correctness[(COPY)<model_name>.jsonl]
