from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset
import random
import time

# Load the HumanEval dataset
dataset = load_dataset("openai_humaneval")

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Check for GPU availability
if not torch.cuda.is_available():
    raise EnvironmentError("No GPU found. Ensure a GPU is available and properly configured.")
device = torch.device("cuda:1")  # Use the specified GPU
print(f"Using device: {device}")

# Enable GPU memory optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Tensor Cores for faster matrix multiplications

# Load the LLaMA model and tokenizer with Accelerate's device management
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired LLaMA model
print(f"Loading model {model_name}...")

# Load tokenizer and model with token-based authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
    device_map="auto",  # Automatically manage device placement
    offload_folder="offload"  # For disk-based offloading if needed
)

# Confirm model loaded
print("Model loaded successfully!")

# Function to mask continuous words
def mask_continuous_words(code, mask_ratio=0.1):
    words = code.split()
    total_words = len(words)
    num_to_mask = max(1, int(round(total_words * mask_ratio)))
    start_index = random.randint(0, total_words - num_to_mask)
    for i in range(start_index, start_index + num_to_mask):
        words[i] = ''  # Replace with an empty string
    return ' '.join(words)

# Function to generate code completion
def generate_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=512)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

# Generate completions for each task in HumanEval
completions = []
total_time = 0  # Track total time
num_tasks = len(dataset["test"])
print("Generating completions...")

for task in dataset["test"]:
    # Use the prompt for the task
    prompt = task["prompt"]
    canonical_solution = task["canonical_solution"]
    
    # Generate masked code
    masked_code = mask_continuous_words(canonical_solution)
    
    # Create the final prompt
    final_prompt = (
        "Please complete the following incomplete code to match the original solution. "
        "Do not add any extra code or function definitions. Only return the completed code, "
        "without any comments or explanations.\n\n"
        f"Here is the code:\n\n```{prompt}+{masked_code}```\n\n"
        "Please provide the completed code:"
    )
    
    # Measure time for generation
    start_time = time.time()
    completion = generate_completion(final_prompt)
    end_time = time.time()
    
    # Calculate time taken
    time_taken = end_time - start_time
    total_time += time_taken
    print(f"Task ID {task['task_id']} took {time_taken:.2f} seconds")
    
    # Append task ID, prompt, masked code, and generated completion
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "masked_code": masked_code,
        "completion": completion,
        "time_taken": time_taken
    })

# Calculate average time
average_time = total_time / num_tasks
print(f"Average time per task: {average_time:.2f} seconds")

# Save completions to a JSONL file
import json
output_file = f"samples_{model_name.replace('/', '_')}.jsonl"
with open(output_file, "w") as f:
    for item in completions:
        json.dump(item, f)
        f.write("\n")

print(f"Completions saved to {output_file}")



# ===========================================================

# Evaluate the model on HumanEval
# python human_eval/evaluate_functional_correctness.py samples.jsonl

# ===========================================================

