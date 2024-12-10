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

# Check for GPU availability
if not torch.cuda.is_available():
    raise EnvironmentError("No GPU found. Ensure a GPU is available and properly configured.")
device = torch.device("cuda")  # Use the first GPU
print(f"Using device: {device}")

# Enable GPU memory optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Tensor Cores for faster matrix multiplications

# Load the LLaMA model and tokenizer with Accelerate's device management
model_name = "meta-llama/Llama-3.1-70B-Instruct"  # Replace with your desired LLaMA model
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

# Function to generate code completion
def generate_completion(prompt):
    """
    Generate code completion using the model.
    Args:
        prompt (str): Input code prompt for the model.
    Returns:
        str: Generated completion text.
    """
    # Tokenize the input and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate output with max tokens specified
    outputs = model.generate(inputs.input_ids, max_new_tokens=512)
    # Decode the generated tokens to text
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion

# Generate completions for each task in HumanEval
completions = []
print("Generating completions...")
for task in dataset["test"]:
    # Use the prompt for the task
    prompt = task["prompt"]
    # Generate completion
    completion = generate_completion(prompt)
    # Append task ID and generated completion
    completions.append({
        "task_id": task["task_id"],
        "prompt": prompt,
        "completion": completion
    })

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

