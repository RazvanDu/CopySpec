import re
import os
import json
import time
import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm  # Used for displaying progress bars
from sympy import sympify, simplify

# =======================
# Utility Functions
# =======================
def extract_boxed_answer(text):
    """
    Extract the content of the last \boxed{...} from the generated model output.
    - Supports multi-line matching.
    - Filters out placeholder 'answer'.
    """
    pattern = re.compile(r'\\boxed\{([\s\S]*?)\}', re.DOTALL)
    all_matches = pattern.findall(text)
    filtered = [m for m in all_matches if m.strip().lower() != 'answer']
    return filtered[-1].strip() if filtered else None


def extract_gold_answer(text):
    """
    Extract the answer following `####` from the gold_answer.
    """
    match = re.search(r"####\s*([^\n]+)", text)
    return match.group(1).strip() if match else None


def evaluate_prediction(pred_answer, gold_answer):
    """
    Compare the predicted answer with the gold answer, supporting fractions, floats, and mathematical expressions.
    """
    if not pred_answer:
        return False

    pred_answer = pred_answer.strip()
    gold_answer = gold_answer.strip()

    try:
        # Try parsing the predicted and gold answers as floats (or simplifiable mathematical expressions)
        pred_expr = simplify(sympify(pred_answer))  
        gold_expr = simplify(sympify(gold_answer))

        pred_value = float(pred_expr)
        gold_value = float(gold_expr)

        # Use float comparison with precision control
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)
    except Exception:
        # Fallback to direct string comparison if parsing fails
        return pred_answer == gold_answer


# =======================
# Multi-Round Reasoning Function
# =======================
def multi_round_cot(input_question, model, tokenizer, device, max_new_tokens=256):
    """
    Implements a three-round reasoning process:
    - Round 1: Preliminary answer with step-by-step reasoning prompted by COT.
    - Round 2: Self-review with problem analysis prompted by COT.
    - Round 3: Improved answer with optimized reasoning prompted by COT.
    """
    def prepare_input(text):
        encoded = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        for k, v in encoded.items():
            encoded[k] = v.to(device)
        return encoded

    total_new_tokens = 0
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # ---------- Round 1 ----------
    prompt_cot_1 = (
        f"Question: {input_question}\n\n"
        "Let's solve this step by step. Start with analyzing the problem carefully.\n"
        "Provide a detailed explanation and put your final answer in the form of \\boxed{answer}.\n"
    )
    inputs_1 = prepare_input(prompt_cot_1)
    start_time_r1 = time.time()
    outputs_1 = model.generate(**inputs_1, generation_config=generation_config)
    time_r1 = time.time() - start_time_r1

    new_tokens_r1 = len(outputs_1[0]) - len(inputs_1["input_ids"][0])
    total_new_tokens += max(new_tokens_r1, 0)

    cot_round_1 = tokenizer.decode(outputs_1[0], skip_special_tokens=True)
    answer_round_1 = extract_boxed_answer(cot_round_1)

    # ---------- Round 2 ----------
    prompt_cot_2 = (
        f"Question: {input_question}\n\n"
        f"Your initial answer:\n{cot_round_1}\n\n"
        "Let's carefully review your reasoning step by step. "
        "Identify any mistakes or areas of improvement, and write a detailed analysis.\n"
    )
    inputs_2 = prepare_input(prompt_cot_2)
    start_time_r2 = time.time()
    outputs_2 = model.generate(**inputs_2, generation_config=generation_config)
    time_r2 = time.time() - start_time_r2

    new_tokens_r2 = len(outputs_2[0]) - len(inputs_2["input_ids"][0])
    total_new_tokens += max(new_tokens_r2, 0)

    cot_round_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)

    # ---------- Round 3 ----------
    prompt_cot_3 = (
        f"Question: {input_question}\n\n"
        f"Based on your analysis in Round 2:\n{cot_round_2}\n\n"
        "Let's refine your reasoning step by step and provide the final improved answer. "
        "Reiterate your reasoning and present your final answer in the form of \\boxed{answer}.\n"
    )
    inputs_3 = prepare_input(prompt_cot_3)
    start_time_r3 = time.time()
    outputs_3 = model.generate(**inputs_3, generation_config=generation_config)
    time_r3 = time.time() - start_time_r3

    new_tokens_r3 = len(outputs_3[0]) - len(inputs_3["input_ids"][0])
    total_new_tokens += max(new_tokens_r3, 0)

    cot_round_3 = tokenizer.decode(outputs_3[0], skip_special_tokens=True)
    answer_round_3 = extract_boxed_answer(cot_round_3)

    total_time = time_r1 + time_r2 + time_r3

    return {
        "round_1": cot_round_1.strip(),
        "answer_round_1": answer_round_1,  # Extracted answer from Round 1
        "round_2": cot_round_2.strip(),
        "round_3": cot_round_3.strip(),
        "final_answer": answer_round_3,    # Final answer extracted from Round 3
        "total_time_s": total_time,
        "total_tokens": total_new_tokens
    }

# =======================
# Main Function for Dataset Execution
# =======================
def run_on_full_dataset(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="gsm8k_results",
    chunk_size=1000,
    max_print_per_split=2,
    test_only=True
):
    """
    Run the dataset and calculate TPS for each round while displaying a progress bar.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("gsm8k", "main")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Determine splits
    split_names = ["test"] if test_only else ["train", "test"]

    for split_name in tqdm(split_names, desc="Splits"):  # First-level tqdm: shows dataset split progress
        split_data = dataset[split_name]

        # Initialize accumulators
        accumulated_tokens_r1, accumulated_time_r1 = 0, 0
        accumulated_tokens_r2, accumulated_time_r2 = 0, 0
        accumulated_tokens_r3, accumulated_time_r3 = 0, 0

        # tqdm for sample processing progress
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", leave=False)):
            question = sample["question"]
            full_gold_answer = sample["answer"]

            # Call the multi-round reasoning function
            result = multi_round_cot(question, model, tokenizer, device)

            # Accumulate tokens and time for each round
            accumulated_tokens_r1 += result["tokens_r1"]
            accumulated_time_r1 += result["time_r1"]

            accumulated_tokens_r2 += result["tokens_r2"]
            accumulated_time_r2 += result["time_r2"]

            accumulated_tokens_r3 += result["tokens_r3"]
            accumulated_time_r3 += result["time_r3"]

        # Calculate TPS for each round
        tps_r1 = accumulated_tokens_r1 / accumulated_time_r1 if accumulated_time_r1 > 0 else 0
        tps_r2 = accumulated_tokens_r2 / accumulated_time_r2 if accumulated_time_r2 > 0 else 0
        tps_r3 = accumulated_tokens_r3 / accumulated_time_r3 if accumulated_time_r3 > 0 else 0

        # Print results
        print(f"[{split_name} Results]")
        print(f" - Round 1 TPS: {tps_r1:.2f} tokens/s")
        print(f" - Round 2 TPS: {tps_r2:.2f} tokens/s")
        print(f" - Round 3 TPS: {tps_r3:.2f} tokens/s")

if __name__ == "__main__":
    run_on_full_dataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="gsm8k_results",
        test_only=True
    )