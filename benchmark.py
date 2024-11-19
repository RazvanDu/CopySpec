import time
from speculative_decoding import SpeculativeDecoder
from typing import Tuple
import torch

def benchmark(
    main_model_name: str,
    draft_model_name: str,
    top_p, # Feel free to pick either top_p or top_k.
    top_k,
    max_tokens) -> Tuple[float, float]:
    """ Runs speculative decoding and compare it against greedy decoding with
    the main model.

    Returns:
      A tuple of (spec_decoding_runtime, greedy_decoding_runtime)
    """
    # TODO: Implement.
    decoder = SpeculativeDecoder(main_model_name, draft_model_name)
    
    prompt = "Once upon a time,"
    
    start_time = time.time()
    torch.manual_seed(123)
    greedy_output = decoder.target_generate_greedy(
        prompt,
        max_new_tokens=max_tokens,
    )

    greedy_decoding_runtime = time.time() - start_time

    start_time = time.time()
    torch.manual_seed(123)
    spec_output = decoder.generate(
        prompt,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        gamma=5,
        max_new_tokens=max_tokens
    )
    spec_decoding_runtime = time.time() - start_time
    
    print("Speculative Decoding Output:")
    print(spec_output)
    print("\nGreedy Decoding Output:")
    print(greedy_output)
    
    return (spec_decoding_runtime, greedy_decoding_runtime)

def calculate_tokens_per_second(num_tokens: int, runtime: float) -> float:
    """
    Calculate the number of tokens generated per second.

    Args:
        num_tokens (int): The number of tokens generated.
        runtime (float): The time taken to generate the tokens, in seconds.

    Returns:
        float: The number of tokens generated per second.
    """
    return num_tokens / runtime if runtime > 0 else 0

def benchmark_v2(
    main_model_name: str,
    draft_model_name: str,
    top_p: float,
    top_k: int,
    max_tokens: int,
    num_runs: int = 1
) -> Tuple[float, float, float, float]:
    """
    Runs speculative decoding and compares it against greedy decoding with the main model.

    Args:
        main_model_name (str): Name of the main (larger) model.
        draft_model_name (str): Name of the draft (smaller) model.
        top_p (float): The cumulative probability threshold for top-p sampling.
        top_k (int): The number of top tokens to consider for top-k sampling.
        max_tokens (int): The maximum number of new tokens to generate.
        num_runs (int): Number of runs to average over. Defaults to 5.

    Returns:
        A tuple of (avg_spec_decoding_runtime, avg_greedy_decoding_runtime, 
                    avg_spec_tokens_per_second, avg_greedy_tokens_per_second)
    """
    decoder = SpeculativeDecoder(main_model_name, draft_model_name)
    prompt = """
    "This is just a random sentence that we use to test that the technique works, it doesn't matter what is written here."

    Please write the first sentence exactly as it is written: 
    """

    spec_runtimes = []
    greedy_runtimes = []
    spec_tokens_per_second = []
    greedy_tokens_per_second = []

    for _ in range(num_runs):
        # Benchmark speculative decoding
        start_time = time.time()
        torch.manual_seed(123)
        spec_output = decoder.generate(
            prompt,
            temperature=0.0,
            top_k=top_k,
            top_p=top_p,
            gamma=10,
            max_new_tokens=max_tokens
        )
        spec_runtime = time.time() - start_time
        spec_runtimes.append(spec_runtime)
        
        spec_tokens = len(decoder.tokenizer.encode(spec_output)) - len(decoder.tokenizer.encode(prompt))
        spec_tokens_per_second.append(calculate_tokens_per_second(spec_tokens, spec_runtime))

        # Benchmark target greedy decoding
        start_time = time.time()
        torch.manual_seed(123)
        greedy_output = decoder.target_generate_greedy(
            prompt,
            max_new_tokens=max_tokens,
        )
        greedy_runtime = time.time() - start_time
        greedy_runtimes.append(greedy_runtime)
        
        greedy_tokens = len(decoder.tokenizer.encode(greedy_output)) - len(decoder.tokenizer.encode(prompt))
        greedy_tokens_per_second.append(calculate_tokens_per_second(greedy_tokens, greedy_runtime))

    avg_spec_runtime = sum(spec_runtimes) / num_runs
    avg_greedy_runtime = sum(greedy_runtimes) / num_runs
    avg_spec_tokens_per_second = sum(spec_tokens_per_second) / num_runs
    avg_greedy_tokens_per_second = sum(greedy_tokens_per_second) / num_runs

    print(f"\nAverage over {num_runs} runs:")
    print(f"Speculative Decoding Runtime: {avg_spec_runtime:.2f} seconds")
    print(f"Greedy Decoding Runtime: {avg_greedy_runtime:.2f} seconds")
    print(f"Speedup: {avg_greedy_runtime / avg_spec_runtime:.2f}x")
    print(f"Speculative Decoding Tokens/second: {avg_spec_tokens_per_second:.2f}")
    print(f"Greedy Decoding Tokens/second: {avg_greedy_tokens_per_second:.2f}")

    # Print sample outputs for verification
    print("\nSample Speculative Decoding Output:")
    print(spec_output)
    print("\nSample Greedy Decoding Output:")
    print(greedy_output)

    return (avg_spec_runtime, avg_greedy_runtime, avg_spec_tokens_per_second, avg_greedy_tokens_per_second)
    
if __name__ == '__main__':
    
    main_model = "gpt2-medium"
    draft_model = "distilgpt2"
    top_p = 1
    top_k = 0
    max_tokens = 200

    # spec_time, greedy_time = benchmark(main_model, draft_model, top_p, top_k, max_tokens)
    # print(f"\nSpeculative Decoding Runtime: {spec_time:.2f} seconds")
    # print(f"Greedy Decoding Runtime: {greedy_time:.2f} seconds")
    # print(f"Speedup: {greedy_time / spec_time:.2f}x")
    # to get avg over multiple runs, un-comment below line
    benchmark_v2(main_model, draft_model, top_p, top_k, max_tokens)

    