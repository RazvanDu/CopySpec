#!/usr/bin/env python
# coding: utf-8
import argparse
import sys

# Add your custom path
custom_path = "/mnt/razvandu/speculative_decoding/"
if custom_path not in sys.path:
    sys.path.append(custom_path)

from speculative_copying import SpeculativeDecoder
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import time
import evaluate

def main():
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

    eos_token = decoder.tokenizer.eos_token

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    
    def extractive_prompt(article_text):
        return (
            #"You are a helpful assistant. "
            "Please produce an *extractive* summary of the article below by choosing "
            "2 or 3 key sentences from the original text:\n\n"
            f"{article_text}\n\n"
            "Return only sentences from the original text that best capture the main ideas. Only write the summary and nothing else"
        )

    # 5. Prepare for evaluation
    predictions = []
    references = []

    total_time = 0
    total_tokens = 0

    # We'll use tqdm to show progress
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        article = sample["article"]
        gold_summary = sample["highlights"]

        prompt = extractive_prompt(article)
        #output = summarizer(prompt, max_length=128, do_sample=False)[0]["generated_text"]

        messages = [
            {"role": "user", "content": prompt}
        ]

        prompt = decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        start_time_r1 = time.time()
        if use_copy:
            output = decoder.generate(
                prompt,
                temperature=0.0,
                top_k=0,
                top_p=1,
                gamma=gamma,
                max_new_tokens=max_new_tokens
            )[0]
            accepted_r1 = accepted_r1 + decoder.total_accepted
        else:
            output = decoder.target_generate_greedy(
                prompt,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
            )
        time_r1 = time.time() - start_time_r1

        total_time += time_r1
        
        output = output[len(prompt):-len(eos_token)]
        tokens_generated_1 = len(decoder.tokenizer.encode(output))
        total_tokens += tokens_generated_1

        messages.append({"role": "assistant", "content": output})

        print(idx, ": ", decoder.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        predictions.append(output)
        references.append(gold_summary)

        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=predictions, references=references)

        print("\n=== ROUGE Scores ===")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        tps = total_tokens/total_time
        print(f"TPS: {tps:.2f} tokens/s\n")
        if idx == 100:
            break

if __name__ == "__main__":
    main()
