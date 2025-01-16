import re
import os
import json
import time
import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm  # for displaying progress bars
from sympy import sympify, simplify
from speculative_copying import SpeculativeDecoder

# Load the LLaMA model and tokenizer with authentication
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired LLaMA model
top_p = 1
top_k = 0
max_new_token = 300

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder = SpeculativeDecoder(model_name, "openai-community/gpt2", device=device)

# =======================
# Utility functions
# =======================
def extract_boxed_answer(text):
    """
    Extract the last \boxed{...} content from the model's generated output.
    - Support multi-line matching.
    - Filter out the 'answer' placeholder.
    """
    pattern = re.compile(r'\\boxed\{([\s\S]*?)\}', re.DOTALL)
    all_matches = pattern.findall(text)
    filtered = [m for m in all_matches if m.strip().lower() != 'answer']
    return filtered[-1].strip() if filtered else None


def extract_gold_answer(text):
    """
    Extract the answer after `####` from gold_answer.
    """
    match = re.search(r"####\s*([^\n]+)", text)
    return match.group(1).strip() if match else None


def evaluate_prediction(pred_answer, gold_answer):
    """
    Compare the predicted answer with the standard answer, supporting fractions, floats, and mathematical expressions.
    """
    if not pred_answer:
        return False

    pred_answer = pred_answer.strip()
    gold_answer = gold_answer.strip()

    try:
        # Attempt to parse the predicted answer and the standard answer as floats (or simplifiable mathematical expressions)
        pred_expr = simplify(sympify(pred_answer))  
        gold_expr = simplify(sympify(gold_answer))

        pred_value = float(pred_expr)
        gold_value = float(gold_expr)

        # Use floating-point comparison, supporting precision control
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)
    except Exception:
        # If parsing fails, fall back to direct string comparison
        return pred_answer == gold_answer


# =======================
# Multi-round reasoning function
# =======================
def multi_round_cot(input_question, decoder, max_new_tokens=256):
    """
    Implement the three-round reasoning process:
    - Round 1: Initial answer.
    - Round 2: Self-review.
    - Round 3: Revised answer.
    """
    def prepare_prompt(text):
        return text

    total_new_tokens = 0

    # ---------- Round 1 ----------
    prompt_cot_1 = (
        f"Question: {input_question}\n\n"
        "Can you solve the following math problem?\n"
        "Please put your answer in the form of \\boxed{answer}.\n"
    )
    start_time_r1 = time.time()
    outputs_1 = decoder.generate(
        prepare_prompt(prompt_cot_1),
        temperature=0.0,
        top_k=0,
        top_p=1,
        gamma=5,
        max_new_tokens=max_new_tokens
    )
    time_r1 = time.time() - start_time_r1

    cot_round_1 = outputs_1[0]
    answer_round_1 = extract_boxed_answer(cot_round_1)

    # ---------- Round 2 ----------
    prompt_cot_2 = (
        f"Question: {input_question}\n\n"
        f"Previous answer:\n{cot_round_1}\n\n"
        "Review your previous answer and find problems with your answer.\n"
    )
    start_time_r2 = time.time()
    outputs_2 = decoder.generate(
        prepare_prompt(prompt_cot_2),
        temperature=0.0,
        top_k=0,
        top_p=1,
        gamma=5,
        max_new_tokens=max_new_tokens
    )
    time_r2 = time.time() - start_time_r2

    cot_round_2 = outputs_2[0]

    # ---------- Round 3 ----------
    prompt_cot_3 = (
        f"Question: {input_question}\n\n"
        f"Revised reasoning:\n{cot_round_2}\n\n"
        "Based on the problems you found, improve your answer. "
        "Please reiterate your answer, with your final answer a single numerical number, "
        "in the form \\boxed{answer}.\n"
    )
    start_time_r3 = time.time()
    outputs_3 = decoder.generate(
        prepare_prompt(prompt_cot_3),
        temperature=0.0,
        top_k=0,
        top_p=1,
        gamma=5,
        max_new_tokens=max_new_tokens
    )
    time_r3 = time.time() - start_time_r3

    cot_round_3 = outputs_3[0]
    answer_round_3 = extract_boxed_answer(cot_round_3)

    total_time = time_r1 + time_r2 + time_r3

    return {
        "round_1": cot_round_1.strip(),
        "answer_round_1": answer_round_1,  # Round1 extracted answer
        "round_2": cot_round_2.strip(),
        "round_3": cot_round_3.strip(),
        "final_answer": answer_round_3,    # Round3 extracted final answer
        "total_time_s": total_time,
    }

def run_on_full_dataset(
    decoder,
    output_dir="gsm8k_results",
    chunk_size=1000,
    max_print_per_split=2,
    test_only=True
):
    """
    Main function:
    - Load the dataset.
    - Multi-round reasoning.
    - Evaluate (separately count Round1 / Round3 accuracy and TPS), and save the results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")  # which includes train/test
    print("Dataset loaded.")

    split_names = ["test"] if test_only else ["train", "test"]

    for split_name in tqdm(split_names, desc="Splits"):
        split_data = dataset[split_name]
        print(f"Processing split: {split_name} ({len(split_data)} samples)")

        results_buffer = []
        file_index = 0

        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", leave=False)):
            question = sample["question"]
            full_gold_answer = sample["answer"]
            gold_answer = extract_gold_answer(full_gold_answer)

            if gold_answer is None:
                print(f"Warning: No valid gold answer in sample {idx}. Skipping.")
                continue

            # ---------- Multi-round reasoning ----------
            result = multi_round_cot(question, decoder)

            # ---------- Collect info to buffer ----------
            results_buffer.append({
                "question": question,
                "gold_answer": full_gold_answer,
                "predicted_answer_round1": result["answer_round_1"],
                "predicted_answer_round3": result["final_answer"],
            })

            if len(results_buffer) >= chunk_size:
                filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(results_buffer, f, indent=2)
                results_buffer, file_index = [], file_index + 1

        if results_buffer:
            filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_buffer, f, indent=2)

    print("Done!")

# =======================
# Main function for running on the dataset
# =======================
def run_on_full_dataset(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="gsm8k_results",
    chunk_size=1000,
    max_print_per_split=2,
    test_only=True
):
    """
    Main function:
    - Load the dataset.
    - Multi-round reasoning.
    - Evaluate (separately count Round1 / Round3 accuracy and TPS), and save the results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")  # which includes train/test
    print("Dataset loaded.")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="sequential",
        offload_folder="offload"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Model loaded.")

    # If you only want to run test, only process test; otherwise process train + test
    split_names = ["test"] if test_only else ["train", "test"]

    for split_name in tqdm(split_names, desc="Splits"):
        split_data = dataset[split_name]
        print(f"Processing split: {split_name} ({len(split_data)} samples)")

        # Buffer for chunked storage
        results_buffer = []
        file_index = 0

        # Accumulated stats (for accuracy and TPS)
        accumulated_correct_round1 = 0
        accumulated_correct_round3 = 0
        accumulated_samples = 0
        accumulated_tokens = 0.0
        accumulated_time = 0.0

        # In the main loop of run_on_full_dataset, add debug print for all rounds:
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", leave=False)):
            question = sample["question"]
            full_gold_answer = sample["answer"]
            gold_answer = extract_gold_answer(full_gold_answer)

            if gold_answer is None:
                print(f"Warning: No valid gold answer in sample {idx}. Skipping.")
                continue

            # ---------- Multi-round reasoning ----------
            result = multi_round_cot(question, model, tokenizer, device)

            # ---------- Optional: debug print ----------
            if idx < max_print_per_split:
                print(f"\n--- Sample {idx} ---")
                print("Question:", question)
                print("Gold Answer:", full_gold_answer)
                print("Extracted Gold Answer:", gold_answer)
                print("\n--- Model Outputs ---")
                print("[Round 1 Output]")
                print(result["round_1"])
                print("[Extracted Answer (Round 1)]")
                print(result["answer_round_1"])
                print("\n[Round 2 Output]")
                print(result["round_2"])
                print("\n[Round 3 Output]")
                print(result["round_3"])
                print("[Extracted Answer (Round 3)]")
                print(result["final_answer"])
                print("=" * 70)

            # ---------- Evaluate correctness of Round1 / Round3 ----------
            is_correct_r1 = evaluate_prediction(result["answer_round_1"], gold_answer)
            is_correct_r3 = evaluate_prediction(result["final_answer"], gold_answer)

            # ---------- Accumulate correct counts, sample count, token count, and time ----------
            accumulated_correct_round1 += int(is_correct_r1)
            accumulated_correct_round3 += int(is_correct_r3)
            accumulated_samples += 1
            accumulated_tokens += result["total_tokens"]
            accumulated_time += result["total_time_s"]

            # ---------- Collect info to buffer, write to file by chunk_size ----------
            results_buffer.append({
                "question": question,
                "gold_answer": full_gold_answer,
                "extracted_gold_answer": gold_answer,
                "predicted_answer_round1": result["answer_round_1"],
                "predicted_answer_round3": result["final_answer"],
                "is_correct_round1": is_correct_r1,
                "is_correct_round3": is_correct_r3
            })

            if len(results_buffer) >= chunk_size:
                filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(results_buffer, f, indent=2)
                results_buffer, file_index = [], file_index + 1

        # ---------- Write remaining part ----------
        if results_buffer:
            filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_buffer, f, indent=2)

        # ========== Compute and print stats for this split ==========
        if accumulated_samples > 0:
            acc_r1 = accumulated_correct_round1 / accumulated_samples
            acc_r3 = accumulated_correct_round3 / accumulated_samples
            tps = accumulated_tokens / accumulated_time if accumulated_time > 0 else 0.0

            print(f"\n[{split_name} results]")
            print(f" - Round1 Accuracy: {acc_r1:.4f}  ({accumulated_correct_round1} / {accumulated_samples})")
            print(f" - Round3 Accuracy: {acc_r3:.4f}  ({accumulated_correct_round3} / {accumulated_samples})")
            print(f" - Tokens per Second (TPS): {tps:.2f} tokens/s\n")
        else:
            print(f"No valid samples in {split_name} split.")

    print("Done!")


if __name__ == "__main__":
    run_on_full_dataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="GSM/CopySpec_gsm8k_results",
        chunk_size=1000,
        max_print_per_split=2,
        test_only=True  # Change to False to run train + test
    )