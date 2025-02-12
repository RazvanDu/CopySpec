import re
import os
import json
import math
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from sympy import sympify, simplify
import subprocess
import tempfile
import argparse
import sys

# Add your custom path
custom_path = "/mnt/razvandu/speculative_decoding/"
if custom_path not in sys.path:
    sys.path.append(custom_path)

from speculative_copying import SpeculativeDecoder

# =================================================
# Utility Functions
# =================================================

def extract_code_block(output):
    """
    Extract the last valid Python code block from the output.
    """
    matches = re.findall(r"```python\n(.*?)```", output, re.DOTALL)
    for code in reversed(matches):
        code = code.strip()
        if "print" in code:
            return code
    return ""

def extract_gold_answer(text):
    """
    Extracts the gold answer from the GSM8K dataset.
    """
    match = re.search(r"####\s*([^\n]+)", text)
    return match.group(1).strip() if match else None

def evaluate_prediction(pred_answer, gold_answer):
    """
    Evaluates whether the predicted answer matches the gold answer.
    """
    if not pred_answer:
        return False

    pred_answer = str(pred_answer).strip()
    gold_answer = str(gold_answer).strip()

    try:
        pred_expr = simplify(sympify(pred_answer))
        gold_expr = simplify(sympify(gold_answer))

        pred_value = float(pred_expr)
        gold_value = float(gold_expr)

        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)
    except Exception:
        return pred_answer == gold_answer

def extract_numeric_value(output):
    """
    Extracts a numeric value from a string.
    """
    match = re.search(r"[-+]?\d[\d,]*(\.\d+)?", output)
    if match:
        return float(match.group(0).replace(",", ""))
    return None

def execute_python_code(code):
    """
    Executes Python code and extracts numeric output.
    """
    code = extract_code_block(code)
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_filename = temp_file.name

        result = subprocess.run(
            ["python", temp_filename],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout.strip()
        numeric_value = extract_numeric_value(output)
        return numeric_value, None if result.returncode == 0 else result.stderr.strip()
    except subprocess.TimeoutExpired:
        return None, "Execution timed out."

# =================================================
# Multi-Round Function
# =================================================

def multi_round_program(input_question, decoder, max_new_tokens=1024):

    global accepted_r1
    global accepted_r2
    global accepted_r3
    global gamma
    global delta

    """
    Runs a three-round process to generate, critique, and improve Python code.
    """
    def prepare_input(text, decoder):
        encoded = decoder.tokenizer(
            text,
            return_tensors='pt',
            #padding=True,
            #truncation=True
        )
        for k, v in encoded.items():
            encoded[k] = v.to(device)
        return encoded

    tau_r1 = 0
    tau_r2 = 0
    tau_r3 = 0

    attempted_r1 = 0
    attempted_r2 = 0
    attempted_r3 = 0

    dec_r1 = 0
    dec_r2 = 0
    dec_r3 = 0

    dec_attempted_r1 = 0
    dec_attempted_r2 = 0
    dec_attempted_r3 = 0

    #generation_config = GenerationConfig(
    #    max_new_tokens=max_new_tokens,
    #    do_sample=True,
    #    temperature=0.7,
    #    pad_token_id=tokenizer.eos_token_id
    #)

    start_token = decoder.tokenizer.bos_token

    if start_token is None:
        start_token_len = 0
    else:
        start_token_len = len(start_token)

    # --------------- Round 1 ---------------
    prompt_round_1 = (
        f"Write a Python program formatted in a code block (using triple backticks with the \"python\" specifier) that solves the following problem and prints the solution:\n"
        f"{input_question}\n"
        "Only provide the Python code that prints the output directly without any input from the user. Do not include the question or comments.\n"
        #"The program should directly print the final answer as a number.\n"
    )
    messages = [
    {"role": "user", "content": prompt_round_1}
    ]
    prompt_round_1 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs_1 = prompt_round_1
    start_time_r1 = time.time()
    if use_copy:
        outputs_1 = decoder.generate(
            inputs_1,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            delta=delta,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r1 = accepted_r1 + decoder.total_accepted
    else:
        outputs_1 = decoder.target_generate_greedy(
            inputs_1,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
    time_r1 = time.time() - start_time_r1
    tau_r1 += decoder.turn_tau
    dec_r1 += decoder.dec_tau
    dec_attempted_r1 += decoder.dec_attempted
    attempted_r1 += decoder.turn_attempted
    #outputs_1 = model.generate(**inputs_1, generation_config=generation_config)
    #code_round_1 = extract_code_block(tokenizer.decode(outputs_1[0], skip_special_tokens=True))
    generated_text_1 = outputs_1[start_token_len:]
    code_round_1 = generated_text_1[len(prompt_round_1):]
    tokens_generated_1 = len(decoder.tokenizer.encode(code_round_1))
    #print("ROUND1:", prompt_round_1)
    #print("OUTPUT1:", code_round_1)
    messages.append({"role": "assistant", "content": code_round_1})


    result_round_1, error_round_1 = execute_python_code(code_round_1)

    # --------------- Round 2 ---------------
    #prompt_round_2 = (
    #    f"Here is the Python program generated in Round 1:\n\n```python\n{code_round_1}\n```\n\n"
    #    "Provide a critique of the code. Identify potential issues or improvements. "
    #    "Do not write new code, only feedback."
    #)
    messages.append({"role": "user", "content": "Reiterate over the code and evaluate it to identify potential issues that could lead to an incorrect final answer.\nIf there are none, do not make any modifications.\n"})
    prompt_round_2 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs_2 = prompt_round_2
    start_time_r2 = time.time()
    if use_copy:
        outputs_2 = decoder.generate(
            inputs_2,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            delta=delta,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r2 = accepted_r2 + decoder.total_accepted
    else:
        outputs_2 = decoder.target_generate_greedy(
            inputs_2,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
    time_r2 = time.time() - start_time_r2
    tau_r2 += decoder.turn_tau
    dec_r2 += decoder.dec_tau
    dec_attempted_r2 += decoder.dec_attempted
    attempted_r2 += decoder.turn_attempted
    #outputs_2 = model.generate(**inputs_2, generation_config=generation_config)
    #critique_round_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)

    generated_text_2 = outputs_2[start_token_len:]
    critique_round_2 = generated_text_2[len(prompt_round_2):]
    tokens_generated_2 = len(decoder.tokenizer.encode(critique_round_2))
    messages.append({"role": "assistant", "content": critique_round_2})

    # --------------- Round 3 ---------------
    #prompt_round_3 = (
    #    f"Rewrite the Python program to address the following critique:\n{critique_round_2}\n\n"
    #    "The program should directly print the final answer as a number.\n"
    #    "Only provide the Python code.\n"
    #)
    messages.append({"role": "user", "content": f"Provide the python code of the solution that you consider to be correct.\nKeep the initial version if you consider it to be correct.\n"})
    prompt_round_3 = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs_3 = prompt_round_3
    start_time_r3 = time.time()
    if use_copy:
        outputs_3 = decoder.generate(
            inputs_3,
            temperature=0.0,
            top_k=0,
            top_p=1,
            gamma=gamma,
            delta=delta,
            max_new_tokens=max_new_tokens
        )[0]
        accepted_r3 = accepted_r3 + decoder.total_accepted
    else:
        outputs_3 = decoder.target_generate_greedy(
            inputs_3,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
    time_r3 = time.time() - start_time_r3
    tau_r3 += decoder.turn_tau
    dec_r3 += decoder.dec_tau
    dec_attempted_r3 += decoder.dec_attempted
    attempted_r3 += decoder.turn_attempted
    #outputs_3 = model.generate(**inputs_3, generation_config=generation_config)
    #code_round_3 = extract_code_block(tokenizer.decode(outputs_3[0], skip_special_tokens=True))
    generated_text_3 = outputs_3[start_token_len:]
    code_round_3 = generated_text_3[len(prompt_round_3):]
    messages.append({"role": "assistant", "content": code_round_3})
    tokens_generated_3 = len(decoder.tokenizer.encode(code_round_3))

    final_conversation = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    print("FINAL CONVERSATION:", final_conversation)
    #print("OUTPUT:", code_round_3)

    result_round_3, error_round_3 = execute_python_code(code_round_3)

    return {
        "round_1_code": code_round_1.strip(),
        "result_round_1": result_round_1,
        "error_round_1": error_round_1,
        "round_2_critique": critique_round_2.strip(),
        "round_3_code": code_round_3.strip(),
        "result_round_3": result_round_3,
        "error_round_3": error_round_3,
        "tokens_generated_1": tokens_generated_1,
        "tokens_generated_2": tokens_generated_2,
        "tokens_generated_3": tokens_generated_3,
        "time_r1": time_r1,
        "time_r2": time_r2,
        "time_r3": time_r3,
        "tau_r1": tau_r1,
        "tau_r2": tau_r2,
        "tau_r3": tau_r3,
        "attempted_r1": attempted_r1,
        "attempted_r2": attempted_r2,
        "attempted_r3": attempted_r3,
        "dec_tau_r1": dec_r1,
        "dec_tau_r2": dec_r2,
        "dec_tau_r3": dec_r3,
        "dec_attempted_r1": dec_attempted_r1,
        "dec_attempted_r2": dec_attempted_r2,
        "dec_attempted_r3": dec_attempted_r3,
    }

# =================================================
# Main Function for GSM8K Test Dataset
# =================================================

def run_on_gsm8k_test(
    decoder,
    output_dir="gsm8k_results",
    chunk_size=1000,
    max_print_per_split=2,
    test_only=True,
    max_new_tokens=1024
):
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("gsm8k", "main")  # Load the GSM8K dataset

    global accepted_r1
    global accepted_r2
    global accepted_r3

    test_data = dataset["test"]
    results = []

    correct_1 = 0
    correct_3 = 0

    accumulated_tokens_r1 = 0
    accumulated_tokens_r2 = 0
    accumulated_tokens_r3 = 0
    accumulated_time_r1 = 0.0
    accumulated_time_r2 = 0.0
    accumulated_time_r3 = 0.0
    accumulated_time = 0
    accumulated_tokens = 0

    total_tau_r1 = 0
    total_tau_r2 = 0
    total_tau_r3 = 0

    total_attempted_r1 = 0
    total_attempted_r2 = 0
    total_attempted_r3 = 0

    total_dec_tau_r1 = 0
    total_dec_tau_r2 = 0
    total_dec_tau_r3 = 0

    total_dec_attempted_r1 = 0
    total_dec_attempted_r2 = 0
    total_dec_attempted_r3 = 0

    for idx, sample in enumerate(tqdm(test_data, desc="Processing GSM8K Test Dataset")):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        result = multi_round_program(question, decoder, max_new_tokens=max_new_tokens)

        # Evaluate correctness
        is_correct_r1 = evaluate_prediction(result["result_round_1"], gold_answer)
        is_correct_r3 = evaluate_prediction(result["result_round_3"], gold_answer)

        correct_1 += is_correct_r1
        correct_3 += is_correct_r3

        if result["result_round_1"] != result["result_round_3"]:
            print("HERE!")

        accumulated_tokens_r1 += result["tokens_generated_1"]
        accumulated_tokens_r2 += result["tokens_generated_2"]
        accumulated_tokens_r3 += result["tokens_generated_3"]
        accumulated_tokens = accumulated_tokens_r1 + accumulated_tokens_r2 + accumulated_tokens_r3
        #accumulated_tokens = tokens_total

        accumulated_time_r1 += result["time_r1"]
        accumulated_time_r2 += result["time_r2"]
        accumulated_time_r3 += result["time_r3"]
        accumulated_time = accumulated_time_r1 + accumulated_time_r2 + accumulated_time_r3

        total_tau_r1 += result["tau_r1"]
        total_tau_r2 += result["tau_r2"]
        total_tau_r3 += result["tau_r3"]

        total_attempted_r1 += result["attempted_r1"]
        total_attempted_r2 += result["attempted_r2"]
        total_attempted_r3 += result["attempted_r3"]

        total_dec_tau_r1 += result["dec_tau_r1"]
        total_dec_tau_r2 += result["dec_tau_r2"]
        total_dec_tau_r3 += result["dec_tau_r3"]

        total_dec_attempted_r1 += result["dec_attempted_r1"]
        total_dec_attempted_r2 += result["dec_attempted_r2"]
        total_dec_attempted_r3 += result["dec_attempted_r3"]

        tps = accumulated_tokens / accumulated_time if accumulated_time > 0 else 0.0
        tps_r1 = accumulated_tokens_r1 / accumulated_time_r1 if accumulated_time_r1 > 0 else 0.0
        tps_r2 = accumulated_tokens_r2 / accumulated_time_r2 if accumulated_time_r2 > 0 else 0.0
        tps_r3 = accumulated_tokens_r3 / accumulated_time_r3 if accumulated_time_r3 > 0 else 0.0

        percent_r1 = accepted_r1 / accumulated_tokens_r1 * 100 if accumulated_time_r1 > 0 else 0.0
        percent_r2 = accepted_r2 / accumulated_tokens_r2 * 100 if accumulated_time_r2 > 0 else 0.0
        percent_r3 = accepted_r3 / accumulated_tokens_r3 * 100 if accumulated_time_r3 > 0 else 0.0
        
        acc_r1 = correct_1/(idx+1)
        acc_r3 = correct_3/(idx+1)

        print("RESULTS:", result["result_round_1"], result["result_round_3"], gold_answer)

        print(f" - Round1 Accuracy: {acc_r1:.4f}  ({correct_1} / {idx+1})")
        print(f" - Round3 Accuracy: {acc_r3:.4f}  ({correct_3} / {idx+1})")
        print(f" - Total Tokens (split): {accumulated_tokens}")
        print(f" - Total Time (split): {accumulated_time:.2f} seconds")

        # Print TPS for each round
        print(f"   * Round1 TPS: {tps_r1:.2f} tokens/s")
        print(f"   * Round2 TPS: {tps_r2:.2f} tokens/s")
        print(f"   * Round3 TPS: {tps_r3:.2f} tokens/s\n")
        print(f"   * Total TPS: {tps:.2f} tokens/s\n")

        # Print accepted% for each round
        print(f"   * Round1 accepted: {percent_r1:.2f} %")
        print(f"   * Round2 accepted: {percent_r2:.2f} %")
        print(f"   * Round3 accepted: {percent_r3:.2f} %\n")

        print(f"   * Total tokens round1: {accumulated_tokens_r1}")
        print(f"   * Total tokens round2: {accumulated_tokens_r2}")
        print(f"   * Total tokens round3: {accumulated_tokens_r3}\n")

        print(f"   * Copied/attempted round1: {total_tau_r1}/{total_attempted_r1}")
        print(f"   * Copied/attempted round2: {total_tau_r2}/{total_attempted_r2}")
        print(f"   * Copied/attempted round3: {total_tau_r3}/{total_attempted_r3}\n")

        if total_attempted_r1 > 0:
            print(f"   * Accepted per turn round1: {total_tau_r1/total_attempted_r1}")
            print(f"   * Accepted per turn round2: {total_tau_r2/total_attempted_r2}")
            print(f"   * Accepted per turn round3: {total_tau_r3/total_attempted_r3}\n")

        if total_dec_attempted_r1 > 0:
            print(f"   * Accepted per turn decoding round1: {total_dec_tau_r1/total_dec_attempted_r1}")
        if total_dec_attempted_r2 > 0:
            print(f"   * Accepted per turn decoding round2: {total_dec_tau_r2/total_dec_attempted_r2}")
        if total_dec_attempted_r3 > 0:
            print(f"   * Accepted per turn decoding round3: {total_dec_tau_r3/total_dec_attempted_r3}\n")

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "round_1_code": result["round_1_code"],
            "result_round_1": result["result_round_1"],
            "error_round_1": result["error_round_1"],
            "round_2_critique": result["round_2_critique"],
            "round_3_code": result["round_3_code"],
            "result_round_3": result["result_round_3"],
            "error_round_3": result["error_round_3"],
            "is_correct_round1": is_correct_r1,
            "is_correct_round3": is_correct_r3,
        })

        if idx == 103:
            break

    # Save results
    output_file = os.path.join(output_dir, "gsm8k_test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

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
    draft_name = args.draft_path
    use_copy = args.use_copy
    max_new_tokens = args.max_new_token
    gamma = args.gamma
    delta = args.delta

    # Set the CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = SpeculativeDecoder(model_name, draft_model_name=draft_name, device=device)

    run_on_gsm8k_test(
        decoder=decoder,
        output_dir="GSM/CopySpec_gsm8k_results_COT",
        # chunk_size=1000,
        max_print_per_split=2,
        test_only=True,
        max_new_tokens=max_new_tokens
    )