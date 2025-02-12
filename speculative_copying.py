import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

cache_directory = "place_a_cache_dir_here"

import time

class SpeculativeDecoder:
    """
    Implements speculative decoding for faster text generation using a draft model
    to propose tokens and a target model to verify them.
    """

    def __init__(self, target_model_name, draft_model_name="", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the SpeculativeDecoder with the target and optional draft models.

        Args:
            target_model_name (str): Name of the target model.
            draft_model_name (str, optional): Name of the draft model for speculative decoding.
            device (str, optional): Device to run the models on ('cuda' or 'cpu').
        """

        self.device = device
        self.target_config = config = AutoConfig.from_pretrained(target_model_name, cache_dir=cache_directory, trust_remote_code=True)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name, cache_dir=cache_directory, device_map='auto', trust_remote_code=True)
        if draft_model_name != "":
            self.use_specdec = True
        else:
            self.use_specdec = False
        if self.use_specdec:
            self.draft_config = config = AutoConfig.from_pretrained(draft_model_name, cache_dir=cache_directory)
            self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, cache_dir=cache_directory, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name, cache_dir=cache_directory)
        self.copy_dict = dict()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_model.eval()
        if self.use_specdec:
            self.draft_model.eval()

        self.vocab_size = self.target_model.config.vocab_size

        self.max_context_target = self.target_config.max_position_embeddings
        if self.use_specdec:
            self.max_context_draft = self.draft_config.max_position_embeddings

        self.total_accepted = 0
        self.summed_copied = 0
        self.total_generated = 0

        self.summed_query = 0
        self.total_query = 0

        self.accepted_spec = 0
        self.total_spec = 0

    def preprocess_prompt(self, input_ids, k):
        """
        Precomputes k-gram token hashes from the prompt for efficient copy detection.

        Args:
            input_ids (torch.Tensor): Tokenized input prompt.
            k (int): Length of token sequences to store for potential copying.
        """
        token_ids = input_ids.squeeze(0).tolist() 
        
        for i in range(len(token_ids) - k + 1):
            token_group = tuple(token_ids[i:i + k])  # Extract k consecutive tokens
            token_hash = hash(token_group)  # Hash the group
            
            # Add the hash to the dictionary, recording the position
            if token_hash not in self.copy_dict:
                self.copy_dict[token_hash] = []
            self.copy_dict[token_hash].append(i)  # Append the starting position of the group

    def find_text_position(self, input_ids):
        """
        Finds stored positions of a token sequence based on precomputed hashes.

        Args:
            input_ids (tuple): Tokenized sequence to search for.

        Returns:
            list: List of positions where the sequence appears in the stored prompt.
        """

        # Hash the tokenized group
        token_hash = hash(input_ids)

        # Check if the hash exists in the dictionary
        if token_hash in self.copy_dict:
            return self.copy_dict[token_hash]  # Return the list of positions
        else:
            return []  # Return an empty list if no match is found

    @staticmethod
    def sample(logits, temperature, top_k, top_p):
        """
        Samples from logits using temperature scaling, top-k, and top-p filtering.

        Args:
            logits (torch.Tensor): Logits from the model.
            temperature (float): Softmax temperature for scaling.
            top_k (int): Number of highest-probability tokens to sample from.
            top_p (float): Nucleus sampling threshold.

        Returns:
            torch.Tensor: Probability distribution after filtering.
        """
        if temperature <= 1e-6:
            return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
        
        logits = logits / temperature
        
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return F.softmax(logits, dim=-1)

    def clip_context(self, input_ids, max_dim):
        """
        Clips the input context to a specified length.

        Args:
            input_ids (torch.Tensor): Tokenized input sequence.
            max_dim (int): Maximum allowed length.

        Returns:
            torch.Tensor: Clipped input sequence.
        """
        return input_ids

    def generate_raw(self, prompt, temperature=0.0, top_k=0, top_p=1.0, number_copy=10, gamma=5, delta=5, max_new_tokens=100):
        """
        Generates text using speculative decoding with both target and draft models.

        Args:
            prompt (str): The input text prompt.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling threshold.
            top_p (float): Top-p sampling threshold.
            number_copy (int): Number of tokens to attempt copying.
            gamma (int): Number of tokens used for similarity-based copying.
            delta (int): Number of draft tokens proposed per step.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            tuple: Generated token IDs and count of successfully copied tokens.
        """

        if gamma > 1000:
            self.tau = 1
            self.turn_tau = 1
            self.attempted = 1
            self.turn_attempted = 1

        self.copy_dict = dict()
        stored_gamma = gamma
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        self.preprocess_prompt(prompt_ids, gamma)
        all_token_ids = prompt_ids.squeeze(0).tolist()
        prompt_length = len(all_token_ids)
        total_generated = 0
        target_past_key_values = None
        draft_past_key_values = None

        with torch.no_grad():
            target_outputs = self.target_model(prompt_ids, use_cache=True, return_dict=True, temperature=temperature)
        target_past_key_values = target_outputs.past_key_values

        target_logits = target_outputs.logits
        target_probs = self.sample(target_logits, temperature, top_k, top_p)
        next_token = torch.multinomial(target_probs[:, -1], 1)
        new_token_ids = next_token.squeeze(0).tolist()
        if not isinstance(new_token_ids, list):
            new_token_ids = [new_token_ids]
        all_token_ids.extend(new_token_ids)
        total_generated += len(new_token_ids)
        to_pass_draft = all_token_ids

        self.total_accepted = 0

        self.turn_tau = 0
        self.turn_attempted = 0

        self.dec_tau = 0
        self.dec_attempted = 0

        if not self.use_specdec:
            self.dec_tau = 1
            self.dec_attempted = 1

        stop_token = self.tokenizer.eos_token_id

        draft_kv_cache = None
        
        while True:

            left_to_do = max_new_tokens - total_generated
            gamma = min(gamma, left_to_do)
            if gamma <= 0:
                break

            draft_tokens = None
            draft_probs = None
            new_tokens = None
            copied = False

            if len(all_token_ids) >= gamma:
                last_gamma_tokens = tuple(all_token_ids[-gamma:])
                token_hash = hash(last_gamma_tokens)
                if token_hash in self.copy_dict and self.copy_dict[token_hash][0] + gamma < len(all_token_ids):

                    copied = True
                    first_occurrence = self.copy_dict[token_hash][0]
                    left_tokens = max_new_tokens - total_generated
                    to_add = min(number_copy, left_tokens)

                    draft_chunk = all_token_ids[(first_occurrence + gamma): (first_occurrence + gamma + to_add)]
                    draft_tokens = torch.tensor(draft_chunk, device=self.device).unsqueeze(0)
                    last_token_id = all_token_ids[-1]
                    last_token_tensor = torch.tensor([[last_token_id]], dtype=draft_tokens.dtype, device=draft_tokens.device)
                    draft_tokens = torch.cat([last_token_tensor, draft_tokens], dim=1)

                    if number_copy == 0:
                        draft_tokens = torch.tensor([[last_token_id]], device=draft_tokens.device)

                    vocab_size = self.vocab_size
                    draft_size = draft_tokens.size(1)-1

            if self.use_specdec and not copied:
                
                last_token_id = all_token_ids[-1]
                if not to_pass_draft:
                    input_for_draft = torch.tensor([[last_token_id]], device=self.device)
                else:
                    input_for_draft = torch.tensor([to_pass_draft], device=self.device)

                with torch.no_grad():
                    draft_tokens = []
                    for _ in range(delta):
                        draft_outputs = self.draft_model(
                            input_for_draft, 
                            past_key_values=draft_kv_cache, 
                            use_cache=True, 
                            return_dict=True,
                            temperature=temperature,
                        )
                        draft_kv_cache = draft_outputs.past_key_values
                        draft_logits = draft_outputs.logits
                        draft_probs = self.sample(draft_logits, temperature, top_k, top_p)
                        next_token = torch.multinomial(draft_probs[:, -1], 1)
                        draft_tokens.append(next_token.item())
                        input_for_draft = next_token

                    draft_tokens = torch.tensor([[last_token_id] + draft_tokens], device=self.device)
                    draft_size = draft_tokens.size(1) - 1
                    to_pass_draft = []
                    self.total_spec += delta
                    
            if draft_tokens is not None:

                with torch.no_grad():
                    target_outputs = self.target_model(draft_tokens, use_cache=True, return_dict=True, past_key_values=target_past_key_values, temperature=temperature)
                target_past_key_values = target_outputs.past_key_values
                target_logits = target_outputs.logits
                target_probs = self.sample(target_logits, temperature, top_k, top_p)

                accepted_tokens = []
                broken = False
                for i in range(draft_size):

                    draft_token = draft_tokens[:, i+1]
                    target_prob = target_probs[:, i, draft_token]

                    if target_prob == 1:
                        accepted_tokens.append(draft_token)
                    else:
                        chosen_token = torch.multinomial(target_probs[:, i], 1)[0]
                        accepted_tokens.append(chosen_token)
                        broken = True
                        break

                if not broken:
                    chosen_token = torch.multinomial(target_probs[:, -1], 1)[0]
                    accepted_tokens.append(chosen_token)

                new_tokens = torch.cat(accepted_tokens)

                num_accepted = len(accepted_tokens)

                if copied:
                    self.summed_query += len(accepted_tokens)/draft_tokens.size(1)
                    self.total_query += 1
                    self.total_accepted += num_accepted-1
                    self.turn_attempted += 1
                    self.turn_tau += num_accepted-1
                else:
                    self.dec_tau += num_accepted-1
                    self.dec_attempted += 1

                final_length = len(all_token_ids) + num_accepted - 1
                target_past_key_values = tuple(
                    (key[:, :, :final_length, :], value[:, :, :final_length, :])
                    for key, value in target_past_key_values
                )

                if not copied:

                    if num_accepted == delta+1:
                        if delta != 0:
                            to_pass_draft = to_pass_draft + [accepted_tokens[-2], accepted_tokens[-1]]
                        else:
                            to_pass_draft = to_pass_draft + [accepted_tokens[-1]]
                    else:
                        to_pass_draft = to_pass_draft + [accepted_tokens[-1]]   
                        draft_kv_cache = tuple(
                            (key[:, :, :(final_length-1), :], value[:, :, :(final_length-1), :])
                            for key, value in draft_kv_cache
                        )
                    self.accepted_spec += num_accepted-1
                    
                else:
                    to_pass_draft = to_pass_draft + accepted_tokens
                
            else:
                
                last_token_id = all_token_ids[-1]
                input_for_target = torch.tensor([[last_token_id]], device=self.device)

                with torch.no_grad():
                    target_outputs = self.target_model(input_for_target, use_cache=True, return_dict=True, past_key_values=target_past_key_values, temperature=temperature)
                target_past_key_values = target_outputs.past_key_values
                target_logits = target_outputs.logits
                target_probs = self.sample(target_logits, temperature, top_k, top_p)

                next_token = torch.multinomial(target_probs[:, 0], 1)
                new_tokens = next_token

            new_token_ids = new_tokens.squeeze(0).tolist()

            to_break = False

            if not isinstance(new_token_ids, list):
                new_token_ids = [new_token_ids]

            if stop_token in new_token_ids:
                stop_index = new_token_ids.index(stop_token)  
                new_token_ids = new_token_ids[:(stop_index+1)]
                to_break = True
             
            all_token_ids.extend(new_token_ids)

            total_generated += len(new_token_ids)
            added_size = len(new_token_ids)
            gamma = stored_gamma

            if to_break:
                break

            if gamma > 0:
                for j in range(len(all_token_ids) - added_size - gamma, len(all_token_ids) - gamma):
                    token_group = tuple(all_token_ids[j:j + gamma])
                    token_hash = hash(token_group)
                    start_pos = j
                    if token_hash not in self.copy_dict:
                        self.copy_dict[token_hash] = []
                    if start_pos not in self.copy_dict[token_hash]:
                        self.copy_dict[token_hash].append(start_pos)

            gamma = stored_gamma

        if len(all_token_ids) > max_new_tokens+prompt_length:
            all_token_ids = all_token_ids[0:(max_new_tokens+prompt_length)]

        self.total_generated += len(all_token_ids) - prompt_length
        self.summed_copied += self.total_accepted

        if self.use_specdec and self.total_spec != 0:
            print("We used the draft model for ", self.total_spec, "tokens, accepting", (self.accepted_spec/self.total_spec*100), "out of", delta, "overall.")
        print("So far we accepted", (self.summed_copied/self.total_generated*100), "out of each 100 tokens")
        print("We attempted to copy", self.total_query, "times")
        if self.total_query == 0:
            print("Didn't attempt to copy!")
        else:
            print("Out of those we accepted ", (self.summed_query/self.total_query), "tokens for each 100 tokens")

        all_token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long)
        all_token_ids_tensor = all_token_ids_tensor.unsqueeze(0)

        return all_token_ids_tensor, self.total_accepted

    def generate(self, prompt, temperature=0.0, top_k=0, top_p=1.0, number_copy=10, gamma=5, delta=5, max_new_tokens=100):
        """
        Generates text and decodes the output into human-readable format.

        Args:
            prompt (str): The input text prompt.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling threshold.
            top_p (float): Top-p sampling threshold.
            number_copy (int): Number of tokens to attempt copying.
            gamma (int): Number of tokens used for similarity-based copying.
            delta (int): Number of draft tokens proposed per step.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            tuple: Generated text and count of successfully copied tokens.
        """

        all_token_ids, _ = self.generate_raw(prompt, temperature, top_k, top_p, number_copy, gamma, delta=delta, max_new_tokens=max_new_tokens)

        return self.tokenizer.decode(all_token_ids[0]), self.total_accepted


    def target_generate_greedy_temp(self, prompt, max_new_tokens=50):
        """
        Generates text greedily using the target model with KV caching.

        Args:
            prompt (str): The input text prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Generated text output.
        """
        import torch
        
        # Tokenize the input prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Initial forward pass to get logits and past_key_values
        outputs = self.target_model(**model_inputs, use_cache=True)
        generated_ids = model_inputs["input_ids"]
        
        # Iteratively generate tokens
        for _ in range(max_new_tokens):
            # Greedy decoding: select the token with the highest probability
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            
            # Append the next token to our generated sequence
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            
            # Use the newly generated token along with past_key_values for next step
            outputs = self.target_model(
                input_ids=next_token, 
                past_key_values=outputs.past_key_values, 
                use_cache=True
            )

        # Decode the entire generated sequence
        return self.tokenizer.decode(generated_ids[0])

    def target_generate_greedy_raw(self, prompt, temperature=0.0, top_k=0, top_p=1.0, max_new_tokens=50):
        """
        Generates text one token at a time using greedy decoding.

        Args:
            prompt (str): The input text prompt.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling threshold.
            top_p (float): Top-p sampling threshold.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            torch.Tensor: Generated token sequence.
        """

        tokenss = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        greedy_output = tokenss.squeeze(0).tolist()
        past_key_values = None
        stop_token = self.tokenizer.eos_token_id

        # We have to generate the tokens one by one so that we can use the same sampling function to guarantee the same outputs
        # This doesn't significantly affect performance.
        for _ in range(max_new_tokens):
            with torch.no_grad():
                target_outputs = self.target_model(
                    tokenss,
                    temperature=temperature,
                    return_dict=True,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            past_key_values = target_outputs.past_key_values

            target_logits = target_outputs.logits
            target_probs = self.sample(target_logits, temperature, top_k, top_p)

            next_token = torch.multinomial(target_probs[:, -1], 1)
            greedy_output.append(next_token.item())
            if greedy_output[-1] == stop_token:
                break

            tokenss = torch.tensor([[greedy_output[-1]]], device=self.device)

        greedy_output = torch.tensor([greedy_output], device=self.device)
        return greedy_output

    def target_generate_greedy(self, prompt, temperature=0.0, max_new_tokens=50):
        """
        Generates text using greedy decoding, ensuring the same output format as speculative decoding.

        Args:
            prompt (str): The input text prompt.
            temperature (float): Sampling temperature.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Generated text output.
        """
        self.tau = 1
        self.turn_tau = 1
        self.attempted = 1
        self.turn_attempted = 1
        self.dec_tau = 1
        self.dec_attempted = 1
        return self.tokenizer.decode(self.target_generate_greedy_raw(prompt, max_new_tokens=max_new_tokens, temperature=temperature)[0])