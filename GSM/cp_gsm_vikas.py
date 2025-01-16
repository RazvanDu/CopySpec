import re
import os
import json
import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from sympy import sympify, simplify
import subprocess
import tempfile

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

def multi_round_program(input_question, model, tokenizer, device, max_new_tokens=256):
    """
    Runs a three-round process to generate, critique, and improve Python code.
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

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # --------------- Round 1 ---------------
    prompt_round_1 = (
        f"Write a Python program that solves the following problem and prints the solution:\n"
        f"{input_question}\n"
        "Only provide the Python code. Do not include the question or comments.\n"
        "The program should directly print the final answer as a number.\n"
    )
    inputs_1 = prepare_input(prompt_round_1)
    outputs_1 = model.generate(**inputs_1, generation_config=generation_config)
    code_round_1 = extract_code_block(tokenizer.decode(outputs_1[0], skip_special_tokens=True))

    result_round_1, error_round_1 = execute_python_code(code_round_1)

    # --------------- Round 2 ---------------
    prompt_round_2 = (
        f"Here is the Python program generated in Round 1:\n\n```python\n{code_round_1}\n```\n\n"
        "Provide a critique of the code. Identify potential issues or improvements. "
        "Do not write new code, only feedback."
    )
    inputs_2 = prepare_input(prompt_round_2)
    outputs_2 = model.generate(**inputs_2, generation_config=generation_config)
    critique_round_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=True)

    # --------------- Round 3 ---------------
    prompt_round_3 = (
        f"Rewrite the Python program to address the following critique:\n{critique_round_2}\n\n"
        "The program should directly print the final answer as a number.\n"
        "Only provide the Python code.\n"
    )
    inputs_3 = prepare_input(prompt_round_3)
    outputs_3 = model.generate(**inputs_3, generation_config=generation_config)
    code_round_3 = extract_code_block(tokenizer.decode(outputs_3[0], skip_special_tokens=True))

    result_round_3, error_round_3 = execute_python_code(code_round_3)

    return {
        "round_1_code": code_round_1.strip(),
        "result_round_1": result_round_1,
        "error_round_1": error_round_1,
        "round_2_critique": critique_round_2.strip(),
        "round_3_code": code_round_3.strip(),
        "result_round_3": result_round_3,
        "error_round_3": error_round_3,
    }

# =================================================
# Main Function for GSM8K Test Dataset
# =================================================

def run_on_gsm8k_test(model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("gsm8k", "main")  # Load the GSM8K dataset
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

    test_data = dataset["test"]
    results = []

    for idx, sample in enumerate(tqdm(test_data, desc="Processing GSM8K Test Dataset")):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        result = multi_round_program(question, model, tokenizer, device)

        # Evaluate correctness
        is_correct_r1 = evaluate_prediction(result["result_round_1"], gold_answer)
        is_correct_r3 = evaluate_prediction(result["result_round_3"], gold_answer)

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

    # Save results
    output_file = os.path.join(output_dir, "gsm8k_test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_on_gsm8k_test(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        output_dir="GSM/CopySpec_gsm8k_results_Vikas"
    )