from typing import List
import torch
from speculative_copying import SpeculativeDecoder
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)



class SpeculativeDecoderProvider(DecoderBase):
    def __init__(
        self,
        name: str,
        secondary_model: str = "openai-community/gpt2",
        dataset: str = "",
        force_base_prompt: bool = False,
        instruction_prefix: str = "",
        response_prefix: str = "",
        device_map: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 300,
        **kwargs,
    ):
        """
        A speculative decoding provider for evalplus.

        Parameters:
            name (str): Primary model name.
            secondary_model (str): Secondary model used for speculative decoding.
            dataset (str): Dataset name.
            force_base_prompt (bool): Force base prompt if True.
            instruction_prefix (str): Instruction prefix for chat formatting.
            response_prefix (str): Response prefix for chat formatting.
            device_map (str): Device map for distributed inference.
            dtype (str): Torch dtype for model precision.
            max_new_tokens (int): Maximum tokens to generate.
        """
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.force_base_prompt = force_base_prompt
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        self.total_copy = 0

        print(f"[DEBUG] Initializing SpeculativeDecoderProvider for model: {name} on device: {self.device}")

        # Initialize SpeculativeDecoder
        self.decoder = SpeculativeDecoder(
            target_model_name=name,
            draft_model_name=secondary_model,
            device=self.device,
        )
        print("[DEBUG] SpeculativeDecoder initialized successfully.")

        # Tokenizer and EOS settings
        self.tokenizer = self.decoder.tokenizer        
        
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        print(f"{self.eos = }")

    def is_direct_completion(self) -> bool:
        """
        Check if the task requires direct completion (without chat formatting).
        """
        return self.force_base_prompt or getattr(self.tokenizer, "chat_template", None) is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 1
    ) -> List[str]:
        """
        Generate code using speculative decoding.

        Parameters:
            prompt (str): Input prompt for code generation.
            do_sample (bool): Enable sampling if True.
            num_samples (int): Number of samples to generate (only supports 1 for now).

        Returns:
            List[str]: List of generated code outputs.
        """
        print(f"[DEBUG] SpeculativeDecoderProvider.codegen called with prompt: {prompt[:50]}...")

        if num_samples > 1:
            raise ValueError("[ERROR] SpeculativeDecoderProvider only supports num_samples=1 currently.")

        # Prepare the prompt
        formatted_prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )

        # Generate using speculative decoding
        try:
            generated_text, accepted_tokens = self.decoder.generate(
                formatted_prompt,
                temperature=0.0,
                top_k=0,
                top_p=1,
                gamma=5,
                max_new_tokens=300,
            )
            self.total_copy += accepted_tokens
            print("GGENENENENENE",generated_text)
            print(f"[DEBUG] Generated text length: {len(generated_text)}; Tokens accepted: {accepted_tokens}")
            print(f"[DEBUG] TOTALLLLL Tokens accepted: {self.total_copy}")

        except Exception as e:
            print(f"[ERROR] SpeculativeDecoderProvider.codegen failed: {e}")
            return []

        # Truncate at first EOS token
        eos_index = min(
            (generated_text.find(eos) for eos in self.eos if eos in generated_text),
            default=len(generated_text),
        )
        cleaned_output = generated_text[:eos_index].strip()

        print(f"[DEBUG] Final cleaned output: {cleaned_output[:50]}...")
        print("\n\n")
        print(f"[DEBUG] TOTALLLLL Tokens accepted: {self.total_copy}")
        return [cleaned_output]