from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

# Define models
big_model_name = "facebook/opt-350m"  # Base model for speculative decoding
small_model_name = "facebook/opt-125m"  # Smaller speculative draft model

# Load tokenizer (ensure compatibility between models)
tokenizer = AutoTokenizer.from_pretrained(big_model_name)

# Input prompt
prompt = "Once upon a time, there was a Razvan that "

# Encode input for consistency
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Step 1: Perform Greedy Decoding using Hugging Face Transformers
big_model = AutoModelForCausalLM.from_pretrained(big_model_name)
with torch.no_grad():
    greedy_output = big_model.generate(
        input_ids,
        max_length=50,  # Limit token length
        do_sample=False,  # Greedy decoding
    )
greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

print("Greedy Decoding Output:")
print(greedy_text)

# Step 2: Perform Speculative Decoding with vLLM
# Define sampling parameters for speculative decoding
sampling_params = SamplingParams(
    temperature=0.0,  # Forces deterministic greedy-like behavior
    top_p=1.0,        # Includes all tokens, no top-p cutoff
    max_tokens=50,    # Limit output length
)

# Initialize vLLM for speculative decoding
llm = LLM(
    model=big_model_name,
    speculative_model=small_model_name,
    num_speculative_tokens=5,  # Speculate 5 tokens ahead
    use_v2_block_manager=True,  # Recommended for better performance
)

# Generate speculative decoding output
outputs = llm.generate([prompt], sampling_params)

print("\nSpeculative Decoding Output:")
for output in outputs:
    speculative_text = output.outputs[0].text
    print(speculative_text)

# Step 3: Compare Results
print("\nComparison:")
print(f"Greedy Decoding Matches Speculative Decoding: {greedy_text.strip() == speculative_text.strip()}")
