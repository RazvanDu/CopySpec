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
# Utility functions
# =======================
def extract_boxed_answer(text):
    """
    Extracts the content of the last \boxed{...} from the model's generated output.
    - Supports multiline matching.
    - Filters out the placeholder 'answer'.
    """
    pattern = re.compile(r'\\boxed\{([\s\S]*?)\}', re.DOTALL)
    all_matches = pattern.findall(text)
    filtered = [m for m in all_matches if m.strip().lower() != 'answer']
    return filtered[-1].strip() if filtered else None


def extract_gold_answer(text):
    """
    Extracts the answer from `gold_answer` following `####`.
    """
    match = re.search(r"####\s*([^\n]+)", text)
    return match.group(1).strip() if match else None


def evaluate_prediction(pred_answer, gold_answer):
    """
    Compares the predicted answer with the ground truth.
    - Supports fractions, floating-point numbers, and mathematical expressions.
    """
    if not pred_answer:
        return False

    pred_answer = pred_answer.strip()
    gold_answer = gold_answer.strip()

    try:
        # Attempt to parse the predicted and gold answers as floating-point numbers
        # or simplifiable mathematical expressions
        pred_expr = simplify(sympify(pred_answer))  
        gold_expr = simplify(sympify(gold_answer))

        pred_value = float(pred_expr)
        gold_value = float(gold_expr)

        # Compare floating-point numbers with controlled precision
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)
    except Exception:
        # If parsing fails, fallback to direct string comparison
        return pred_answer == gold_answer


# =======================
# Multi-round inference function
# =======================
def multi_round_cot(input_question, model, tokenizer, device, max_new_tokens=256):
    """
    Implements a three-round reasoning process:
    - Round 1: Initial response using COT (Chain of Thought) prompts for step-by-step reasoning.
    - Round 2: Self-review to analyze reasoning and identify improvements.
    - Round 3: Refine the answer with improved reasoning.
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

# Remaining translation continues in the next message due to length limit.

# =======================
# Main function for dataset processing
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
    - Load dataset.
    - Perform multi-round inference.
    - Evaluate results (accuracy for Round 1 and Round 3, TPS) and save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")  # Contains train/test splits
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

    # Process only test split if specified; otherwise process train + test
    split_names = ["test"] if test_only else ["train", "test"]

    for split_name in tqdm(split_names, desc="Splits"):
        split_data = dataset[split_name]
        print(f"Processing split: {split_name} ({len(split_data)} samples)")

        # Buffer for chunked storage
        results_buffer = []
        file_index = 0

        # Accumulated stats for accuracy and TPS
        accumulated_correct_round1 = 0
        accumulated_correct_round3 = 0
        accumulated_samples = 0
        accumulated_tokens = 0.0
        accumulated_time = 0.0

        # Loop through samples in the split
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", leave=False)):
            question = sample["question"]
            full_gold_answer = sample["answer"]
            gold_answer = extract_gold_answer(full_gold_answer)

            if gold_answer is None:
                print(f"Warning: No valid gold answer in sample {idx}. Skipping.")
                continue

            # ---------- Perform multi-round inference ----------
            result = multi_round_cot(question, model, tokenizer, device)

            # ---------- Optional debugging output ----------
            if idx < max_print_per_split:
                print(f"\n--- Sample {idx} ---")
                print("Question:", question)
                print("Gold Answer:", full_gold_answer)
                print("Extracted Gold Answer:", gold_answer)
                print("\n--- Model Outputs ---")
                print("[Round 1 Output]")
                print(result["round_1"])  # Full output from Round 1
                print("[Extracted Answer (Round 1)]")
                print(result["answer_round_1"])
                print("\n[Round 2 Output]")
                print(result["round_2"])  # Full output from Round 2
                print("\n[Round 3 Output]")
                print(result["round_3"])  # Full output from Round 3
                print("[Extracted Answer (Round 3)]")
                print(result["final_answer"])
                print("=" * 70)

            # ---------- Evaluate correctness for Round 1 and Round 3 ----------
            is_correct_r1 = evaluate_prediction(result["answer_round_1"], gold_answer)
            is_correct_r3 = evaluate_prediction(result["final_answer"], gold_answer)

            # ---------- Accumulate stats ----------
            accumulated_correct_round1 += int(is_correct_r1)
            accumulated_correct_round3 += int(is_correct_r3)
            accumulated_samples += 1
            accumulated_tokens += result["total_tokens"]
            accumulated_time += result["total_time_s"]

            # ---------- Add to buffer and save in chunks ----------
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

        # ---------- Write remaining results to file ----------
        if results_buffer:
            filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_buffer, f, indent=2)

        # ========== Calculate and print stats for this split ==========
        if accumulated_samples > 0:
            acc_r1 = accumulated_correct_round1 / accumulated_samples
            acc_r3 = accumulated_correct_round3 / accumulated_samples
            tps = accumulated_tokens / accumulated_time if accumulated_time > 0 else 0.0

            print(f"\n[{split_name} results]")
            print(f" - Round1 Accuracy: {acc_r1:.4f}  ({accumulated_correct_round1} / {accumulated_samples})")
            print(f" - Round3 Accuracy: {acc_r3:.4f}  ({accumulated_correct_round3} / {accumulated_samples})")
            print(f" - Tokens per Second (TPS): {tps:.2f} tokens/s\n")

    print("Done!")


if __name__ == "__main__":
    run_on_full_dataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="GSM/gsm8k_results_cot",
        chunk_size=1000,
        max_print_per_split=2,
        test_only=True  # Change to False to run both train and test splits
    )