# script_example.py
from speculative_copying import SpeculativeDecoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 你想用的 target 模型
decoder = SpeculativeDecoder(model_name, "openai-community/gpt2", device=device)
print("Speculative decoder loaded successfully!")

def main():
    global_context = ""
    while True:
        user_prompt = input("\nEnter a prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            print("Exiting speculative decoding program.")
            break
        if global_context:
            global_context += " " + user_prompt + "\n"
        else:
            global_context = user_prompt
        
        print("\nGenerating text based on the context...")
        generated_text, total_accepted = decoder.generate(
            global_context,
            temperature=0.0,  
            top_k=0,
            top_p=1,
            max_new_tokens=200,
            gamma=5
        )
        global_context += " " + generated_text
        print(f"\nGenerated Text:\n{generated_text}")
        print(f"Tokens accepted: {total_accepted}")

if __name__ == "__main__":
    main()