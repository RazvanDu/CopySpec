from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def evaluate_model_on_mt_bench(model_name, max_samples=None, device="cuda"):
    """
    Evaluates a Hugging Face model on MT-Bench.

    Args:
        model_name (str): The Hugging Face model identifier.
        max_samples (int): Maximum number of samples to evaluate. If None, evaluates all samples.
        device (str): The device to run the evaluation on, e.g., 'cuda' or 'cpu'.

    Returns:
        List of model outputs and evaluation metrics.
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load the MT-Bench dataset
    dataset = load_dataset("mt-bench")

    # Get the test split
    test_set = dataset["test"]
    if max_samples:
        test_set = test_set.select(range(min(len(test_set), max_samples)))

    results = []

    # Evaluate model on the test set
    for i, example in enumerate(test_set):
        input_text = example["input"]
        expected_output = example["output"]

        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode model output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compare with expected output (simple string matching, could use BLEU, ROUGE, etc.)
        exact_match = generated_text.strip() == expected_output.strip()

        # Store result
        results.append({
            "input": input_text,
            "expected_output": expected_output,
            "generated_output": generated_text,
            "exact_match": exact_match
        })

        print(f"Sample {i+1}: {'MATCH' if exact_match else 'NO MATCH'}")
        print(f"Input: {input_text}")
        print(f"Expected: {expected_output}")
        print(f"Generated: {generated_text}\n")

    # Summary of performance
    exact_match_accuracy = sum(r["exact_match"] for r in results) / len(results)
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}")

    return results


# Example usage
model_name = "gpt2"  # Replace with your model's name on Hugging Face
results = evaluate_model_on_mt_bench(model_name, max_samples=10)
