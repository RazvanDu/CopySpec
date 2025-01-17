import re
import os
import json
import time
import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
from sympy import sympify, simplify
import argparse

# Add your custom path
custom_path = "/mnt/razvandu/speculative_decoding/"
if custom_path not in sys.path:
    sys.path.append(custom_path)

from speculative_copying import SpeculativeDecoder

# =======================
# Utility functions
# =======================
def extract_boxed_answer(text):
    """
    Extracts the content of the last \\boxed{...} from the model's generated output.
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
def multi_round_cot(input_question, decoder, max_new_tokens=1024):
    """
    Implements a three-round reasoning process using SpeculativeDecoder.
    Tracks time and tokens generated for each round.
    Returns detailed info including tokens/time for each round.
    """
    total_new_tokens = 0

    global accepted_r1
    global accepted_r2
    global accepted_r3
    global gamma

    # ---------- Round 1 ----------
    #prompt_cot_1 = (
    #    f"{input_question}\n"
    #    "Give your output in the form of \\boxed{answer}."
    #)
    prompt_cot_1 = (
        "Solve the following problem step-by-step and show all calculations clearly. Present the final answer in the format \\boxed{answer}.\n"
        f"Problem: {input_question}\n"
        #"Give your output in the form of \\boxed{answer}."
    )
    messages = [
    {"role": "user", "content": prompt_cot_1}
    ]
    prompt_cot_1 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("INPUT:", prompt_cot_1)

    start_time_r1 = time.time()
    #print("AAA", max_new_tokens)
    if use_copy:
        outputs_1 = decoder.generate(
            prompt_cot_1,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r1 = accepted_r1 + decoder.total_accepted
    else:
        outputs_1 = decoder.target_generate_greedy(
            prompt_cot_1,
            max_new_tokens=max_new_tokens,
        )[0]
    time_r1 = time.time() - start_time_r1

    start_token = decoder.tokenizer.bos_token

    if start_token is None:
        start_token_len = 0
    else:
        start_token_len = len(start_token)

    #print("OUTPUT:", outputs_1)
    generated_text_1 = outputs_1[start_token_len:]
    generated_text_1 = generated_text_1[len(prompt_cot_1):]
    #print("TRIMMED:", generated_text_1)
    messages.append({"role": "assistant", "content": generated_text_1})
    tokens_generated_1 = len(decoder.tokenizer.encode(generated_text_1, add_special_tokens=False))
    tokens_prompt_1 = len(decoder.tokenizer.encode(prompt_cot_1, add_special_tokens=False))
    tokens_r1 = tokens_generated_1# - tokens_prompt_1
    total_new_tokens += tokens_r1

    #cot_round_1 = generated_text_1[len(prompt_cot_1):]#.strip()
    cot_round_1 = generated_text_1
    print("COT1", cot_round_1)
    answer_round_1 = extract_boxed_answer(cot_round_1)

    # ---------- Round 2 ----------
    #prompt_cot_2 = (
    #    f"Question: {input_question}\n\n"
    #    f"Your initial answer:\n{cot_round_1}\n\n"
    #    "Let's carefully review your reasoning step by step. "
    #    "Identify any mistakes or areas of improvement, and write a detailed analysis.\n"
    #)
    #messages.append({"role": "user", "content": "Review your previous answer and find problems with your answer.\n"})
    messages.append({"role": "user", "content": "Reiterate over your previous solution, carefully reflecting on each step. Identify any errors, inconsistencies, or areas requiring improvement. Highlight any steps that are correct and confirm their validity. Do not recalculate or rewrite the solution; instead, focus on analyzing and commenting on your earlier reasoning.\n"})
    prompt_cot_2 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    start_time_r2 = time.time()
    if use_copy:
        outputs_2 = decoder.generate(
            prompt_cot_2,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r2 = accepted_r2 + decoder.total_accepted
    else:
        outputs_2 = decoder.target_generate_greedy(
            prompt_cot_2,
            max_new_tokens=max_new_tokens,
        )[0]
    time_r2 = time.time() - start_time_r2

    generated_text_2 = outputs_2[start_token_len:]
    #print("OUTPUT2!!!:", generated_text_2)
    generated_text_2 = generated_text_2[len(prompt_cot_2):]
    #print("PARSED2!!!:", generated_text_2)
    messages.append({"role": "assistant", "content": generated_text_2})
    tokens_generated_2 = len(decoder.tokenizer.encode(generated_text_2, add_special_tokens=False))
    tokens_prompt_2 = len(decoder.tokenizer.encode(prompt_cot_2, add_special_tokens=False))
    tokens_r2 = tokens_generated_2# - tokens_prompt_2
    total_new_tokens += tokens_r2

    cot_round_2 = generated_text_2#.strip()

    # ---------- Round 3 ----------
    #prompt_cot_3 = (
    #    f"Question: {input_question}\n\n"
    #    f"Based on your initial answer \n{cot_round_1} and your analysis in Round 2:\n{cot_round_2}\n\n"
    #    "Let's refine your reasoning step by step and provide the final improved answer. "
    #    "Reiterate your reasoning and present your final answer in the form of \\boxed{answer}.\n"
    #)    
    #messages.append({"role": "user", "content": "Analyze the observations above and provide your final answer in the form of \\boxed{answer} even if it stays unchanged.\n"})
    messages.append({"role": "user", "content": "Based on your analysis of the earlier solution, make the necessary corrections to any identified errors. Retain valid steps as they are, and ensure the final answer is accurate. Present the verified answer in the format \\boxed{answer}.\n"})
    prompt_cot_3 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #print("INPUT COT3:", prompt_cot_3)
    start_time_r3 = time.time()
    if use_copy:
        outputs_3 = decoder.generate(
            prompt_cot_3,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r3 = accepted_r3 + decoder.total_accepted
    else:
        outputs_3 = decoder.target_generate_greedy(
            prompt_cot_3,
            max_new_tokens=max_new_tokens,
        )[0]
    time_r3 = time.time() - start_time_r3
    #print("AAAAAAAA", outputs_3)

    generated_text_3 = outputs_3[start_token_len:]
    generated_text_3 = generated_text_3[len(prompt_cot_3):]
    tokens_generated_3 = len(decoder.tokenizer.encode(generated_text_3, add_special_tokens=False))
    tokens_prompt_3 = len(decoder.tokenizer.encode(prompt_cot_3, add_special_tokens=False))
    tokens_r3 = tokens_generated_3# - tokens_prompt_3
    total_new_tokens += tokens_r3

    cot_round_3 = generated_text_3#.strip()
    print("PROMPT COT3:", prompt_cot_3)
    print("COT3", cot_round_3)
    answer_round_3 = extract_boxed_answer(cot_round_3)

    total_time = time_r1 + time_r2 + time_r3

    return {
        "round_1": cot_round_1,
        "answer_round_1": answer_round_1,
        "round_2": cot_round_2,
        "round_3": cot_round_3,
        "final_answer": answer_round_3,
        "total_time_s": total_time,
        "total_tokens": total_new_tokens,
        # Below are additional fields for tracking time and tokens for each round
        "time_r1": time_r1,
        "time_r2": time_r2,
        "time_r3": time_r3,
        "tokens_r1": tokens_r1,
        "tokens_r2": tokens_r2,
        "tokens_r3": tokens_r3,
    }

# =======================
# Main function for dataset processing
# =======================
def run_on_full_dataset(
    decoder,
    output_dir="gsm8k_results",
    chunk_size=1000,
    max_print_per_split=2,
    test_only=True,
    max_new_tokens=1024
):
    """
    Main function:
    - Load dataset.
    - Perform multi-round inference.
    - Evaluate results (accuracy for Round 1 and Round 3, TPS) and save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    print("Dataset loaded.")

    global accepted_r1
    global accepted_r2
    global accepted_r3

    split_names = ["test"] if test_only else ["train", "test"]

    # Global stats for all splits
    total_accumulated_tokens = 0.0
    total_accumulated_time = 0.0

    # Global statistics for each round TPS
    total_accumulated_tokens_r1 = 0.0
    total_accumulated_time_r1 = 0.0
    total_accumulated_tokens_r2 = 0.0
    total_accumulated_time_r2 = 0.0
    total_accumulated_tokens_r3 = 0.0
    total_accumulated_time_r3 = 0.0

    for split_name in tqdm(split_names, desc="Splits"):
        split_data = dataset[split_name]
        print(f"Processing split: {split_name} ({len(split_data)} samples)")

        results_buffer = []
        file_index = 0

        # Current split stats
        accumulated_correct_round1 = 0
        accumulated_correct_round3 = 0
        accumulated_samples = 0
        accumulated_tokens = 0.0
        accumulated_time = 0.0

        # Statistics for each round TPS in the split
        accumulated_tokens_r1 = 0.0
        accumulated_time_r1 = 0.0
        accumulated_tokens_r2 = 0.0
        accumulated_time_r2 = 0.0
        accumulated_tokens_r3 = 0.0
        accumulated_time_r3 = 0.0

        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", leave=False)):
            question = sample["question"]
            full_gold_answer = sample["answer"]
            gold_answer = extract_gold_answer(full_gold_answer)

            if gold_answer is None:
                print(f"Warning: No valid gold answer in sample {idx}. Skipping.")
                continue

            # Perform multi-round inference
            result = multi_round_cot(question, decoder, max_new_tokens=max_new_tokens)

            # Evaluate correctness for Round 1 and Round 3
            print("RESULTS:", result["answer_round_1"], result["final_answer"], gold_answer)
            is_correct_r1 = evaluate_prediction(result["answer_round_1"], gold_answer)
            is_correct_r3 = evaluate_prediction(result["final_answer"], gold_answer)

            # Accumulate stats for the split
            accumulated_correct_round1 += int(is_correct_r1)
            accumulated_correct_round3 += int(is_correct_r3)
            accumulated_samples += 1
            accumulated_tokens += result["total_tokens"]
            accumulated_time += result["total_time_s"]

            # Global stats for all splits
            total_accumulated_tokens += result["total_tokens"]
            total_accumulated_time += result["total_time_s"]

            # Add each round's tokens/time
            accumulated_tokens_r1 += result["tokens_r1"]
            accumulated_time_r1 += result["time_r1"]

            accumulated_tokens_r2 += result["tokens_r2"]
            accumulated_time_r2 += result["time_r2"]

            accumulated_tokens_r3 += result["tokens_r3"]
            accumulated_time_r3 += result["time_r3"]

            # Also add to all splits' overall statistics
            total_accumulated_tokens_r1 += result["tokens_r1"]
            total_accumulated_time_r1 += result["time_r1"]

            total_accumulated_tokens_r2 += result["tokens_r2"]
            total_accumulated_time_r2 += result["time_r2"]

            total_accumulated_tokens_r3 += result["tokens_r3"]
            total_accumulated_time_r3 += result["time_r3"]

            # Save partial results
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

            # Calculate and print stats for the current split
            if accumulated_samples > 0:

                print("Currently at:", idx)

                acc_r1 = accumulated_correct_round1 / accumulated_samples
                acc_r3 = accumulated_correct_round3 / accumulated_samples
                tps = accumulated_tokens / accumulated_time if accumulated_time > 0 else 0.0
                
                # TPS for each round
                tps_r1 = accumulated_tokens_r1 / accumulated_time_r1 if accumulated_time_r1 > 0 else 0.0
                tps_r2 = accumulated_tokens_r2 / accumulated_time_r2 if accumulated_time_r2 > 0 else 0.0
                tps_r3 = accumulated_tokens_r3 / accumulated_time_r3 if accumulated_time_r3 > 0 else 0.0
                tps_overall = (accumulated_tokens_r1 + accumulated_tokens_r2 + accumulated_tokens_r3) / (accumulated_time_r1 + accumulated_time_r2 + accumulated_time_r3) if accumulated_time_r3 > 0 else 0.0

                # Accepted% for each round
                percent_r1 = accepted_r1 / accumulated_tokens_r1 * 100 if accumulated_time_r1 > 0 else 0.0
                percent_r2 = accepted_r2 / accumulated_tokens_r2 * 100 if accumulated_time_r2 > 0 else 0.0
                percent_r3 = accepted_r3 / accumulated_tokens_r3 * 100 if accumulated_time_r3 > 0 else 0.0

                print(f"\n[{split_name} results]")
                print(f" - Round1 Accuracy: {acc_r1:.4f}  ({accumulated_correct_round1} / {accumulated_samples})")
                print(f" - Round3 Accuracy: {acc_r3:.4f}  ({accumulated_correct_round3} / {accumulated_samples})")
                print(f" - Total Tokens (split): {accumulated_tokens}")
                print(f" - Total Time (split): {accumulated_time:.2f} seconds")
                print(f" - Tokens per Second (TPS, split overall): {tps:.2f} tokens/s")

                # Print TPS for each round
                print(f"   * Round1 TPS: {tps_r1:.2f} tokens/s")
                print(f"   * Round2 TPS: {tps_r2:.2f} tokens/s")
                print(f"   * Round3 TPS: {tps_r3:.2f} tokens/s\n")
                print(f"   * Total TPS: {tps_overall:.2f} tokens/s\n")

                # Print accepted% for each round
                print(f"   * Round1 accepted: {percent_r1:.2f} %")
                print(f"   * Round2 accepted: {percent_r2:.2f} %")
                print(f"   * Round3 accepted: {percent_r3:.2f} %\n")

        # Save remaining results for this split
        if results_buffer:
            filename = os.path.join(output_dir, f"{split_name}_part{file_index}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_buffer, f, indent=2)


    #print("\n[Overall results across splits]")
    #print(f" - Total Tokens (all splits): {total_accumulated_tokens}")
    #print(f" - Total Time (all splits): {total_accumulated_time:.2f} seconds")
    #print(f" - Overall Tokens per Second (TPS): {overall_tps:.2f} tokens/s")

    # ========================
    #print(" - Round1 Overall TPS: {:.2f} tokens/s".format(overall_tps_r1))
    #print(" - Round2 Overall TPS: {:.2f} tokens/s".format(overall_tps_r2))
    #print(" - Round3 Overall TPS: {:.2f} tokens/s".format(overall_tps_r3))

    print("Done!")


if __name__ == "__main__":

    # Load the LLaMA model and tokenizer with authentication
    model_name = ""  # Replace with your desired LLaMA model
    top_p = 1
    top_k = 0
    max_new_tokens = 1024
    use_copy = False

    accepted_r1 = 0
    accepted_r2 = 0
    accepted_r3 = 0

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--draft-path",
        type=str,
        default="",
        help="The path to the draft weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--use-copy",
        type=bool,
        default=False,
        help="Whether to use copying or not.",
    )
    
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="The number of tokens to search for.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=5,
        help="The number of tokens that the draft model generates when using speculative decoding",
    )

    args = parser.parse_args()

    model_name = args.model_path
    use_copy = args.use_copy
    max_new_tokens = args.max_new_token
    gamma = args.gamma

    # Set the CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = SpeculativeDecoder(model_name, device=device)

    run_on_full_dataset(
        decoder=decoder,
        output_dir="GSM/CopySpec_gsm8k_results_COT",
        # chunk_size=1000,
        max_print_per_split=2,
        test_only=True,
        max_new_tokens=max_new_tokens
    )