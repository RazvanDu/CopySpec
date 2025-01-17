import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

cache_directory = "/mnt/razvandu/speculative_decoding/models_cache"

import time

class SpeculativeDecoder:
    """
    A class implementing speculative decoding for language models.

    This class uses a larger target model and a smaller draft model to perform
    speculative decoding, potentially speeding up text generation.

    Attributes:
        device (str): The device to run the models on ('cuda' or 'cpu').
        target_model (AutoModelForCausalLM): The larger, more accurate language model.
        draft_model (AutoModelForCausalLM): The smaller, faster language model for draft predictions.
        tokenizer (AutoTokenizer): The tokenizer for both models.
    """

    def __init__(self, target_model_name, draft_model_name="", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the SpeculativeDecoder with target and draft models.

        Args:
            target_model_name (str): The name or path of the target (larger) model.
            draft_model_name (str): The name or path of the draft (smaller) model.
            device (str): The device to run the models on. Defaults to 'cuda' if available, else 'cpu'.
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
        #print(f"Vocabulary size: {vocab_size}")

        self.max_context_target = self.target_config.max_position_embeddings
        if self.use_specdec:
            self.max_context_draft = self.draft_config.max_position_embeddings

        print("TEST", self.max_context_target)

        self.total_accepted = 0
        self.summed_copied = 0
        self.total_generated = 0

        self.summed_query = 0
        self.total_query = 0

        self.accepted_spec = 0
        self.total_spec = 0

        #print("WE CAN AT MOST FIT", self.max_context_size)
        #print(f"Maximum context size: {max_context_size}")

    def preprocess_prompt(self, input_ids, k):
        token_ids = input_ids.squeeze(0).tolist()  # Convert tensor to list of token IDs

        # Iterate over the tokenized prompt to generate groups of k tokens
        for i in range(len(token_ids) - k + 1):
            token_group = tuple(token_ids[i:i + k])  # Extract k consecutive tokens
            #print(token_group)
            token_hash = hash(token_group)  # Hash the group
            
            # Add the hash to the dictionary, recording the position
            if token_hash not in self.copy_dict:
                self.copy_dict[token_hash] = []
            self.copy_dict[token_hash].append(i)  # Append the starting position of the group

    def find_text_position(self, input_ids):
        # Tokenize the input text
        # print("BBB", text)

        #token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device).squeeze(0).tolist()
    
        # print("CCC", self.tokenizer.encode(text, return_tensors="pt").to(self.device))
        # print("AAA", tuple(token_ids))

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
        Adjust logits for sampling based on temperature, top-k, and top-p parameters.

        Args:
            logits (torch.Tensor): The input logits.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top tokens to consider for top-k sampling.
            top_p (float): The cumulative probability threshold for top-p sampling.

        Returns:
            torch.Tensor: The adjusted probability distribution.
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
        #if input_ids.size(1) > max_dim:
        #    input_ids = input_ids[:, -max_dim:]  # Retain the last max_dim tokens
        return input_ids

    def generate_raw(self, prompt, temperature=0.0, top_k=0, top_p=1.0, k=10, gamma=5, delta=5, max_new_tokens=100):

        print("RUNNING WITH GAMMA", gamma, self.use_specdec)

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

        stop_token = self.tokenizer.eos_token_id

        draft_kv_cache = None

        #to_pass_draft = None

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
                    to_add = min(10, left_tokens)

                    draft_chunk = all_token_ids[(first_occurrence + gamma): (first_occurrence + gamma + to_add)]
                    draft_tokens = torch.tensor(draft_chunk, device=self.device).unsqueeze(0)
                    last_token_id = all_token_ids[-1]
                    last_token_tensor = torch.tensor([[last_token_id]], dtype=draft_tokens.dtype, device=draft_tokens.device)
                    draft_tokens = torch.cat([last_token_tensor, draft_tokens], dim=1)
                    
                    vocab_size = self.vocab_size
                    #draft_probs = torch.zeros((draft_tokens.size(1), 1, vocab_size), device=draft_tokens.device)
                    #draft_probs[torch.arange(draft_tokens.size(1)), 0, draft_tokens] = 1
                    #gamma = draft_tokens.size(1)-1
                    draft_size = draft_tokens.size(1)-1

            if self.use_specdec and not copied:
                
                last_token_id = all_token_ids[-1]
                if not to_pass_draft:
                    input_for_draft = torch.tensor([[last_token_id]], device=self.device)
                else:
                    input_for_draft = torch.tensor([to_pass_draft], device=self.device)
                
                #print("!!!INPUT FOR DRAFT:", input_for_draft[0])

                with torch.no_grad():
                    #draft_kv_cache = None
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
                        #draft_probs = draft_logits.softmax(-1)
                        next_token = torch.multinomial(draft_probs[:, -1], 1)
                        draft_tokens.append(next_token.item())
                        #print("AAA", next_token)
                        input_for_draft = next_token
                    draft_tokens = torch.tensor([[last_token_id] + draft_tokens], device=self.device)
                    draft_size = draft_tokens.size(1) - 1
                    to_pass_draft = []
                    self.total_spec += delta
                    
            if draft_tokens is not None:

                #print("DRAFT:", self.tokenizer.decode(draft_tokens[0], skip_special_tokens=True))
                #print("DT", draft_tokens[0])

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

                #print("ACCEPTED:", self.tokenizer.decode(torch.tensor(accepted_tokens), skip_special_tokens=True))
                #print("AT", accepted_tokens)

                new_tokens = torch.cat(accepted_tokens)

                num_accepted = len(accepted_tokens)

                if copied:
                    self.summed_query += len(accepted_tokens)/draft_tokens.size(1)
                    self.total_query += 1
                    self.total_accepted += num_accepted-1

                final_length = len(all_token_ids) + num_accepted - 1
                target_past_key_values = tuple(
                    (key[:, :, :final_length, :], value[:, :, :final_length, :])
                    for key, value in target_past_key_values
                )

                if not copied:
                    #print("A", draft_kv_cache[0][0].shape)
                    #print("B", draft_kv_cache[0][0].shape)
                    #print("@", len(all_token_ids))
                    #print("@@", final_length)
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
                    #print("!!!", self.accepted_spec/self.total_spec, copied)
                else:
                    to_pass_draft = to_pass_draft + accepted_tokens
                    #if to_pass_draft is None:
                    #    to_pass_draft = accepted_tokens
                    #else:
                    #    to_pass_draft += accepted_tokens

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

            #print("")

            #print("CURRENT:", self.tokenizer.decode(torch.tensor(all_token_ids), skip_special_tokens=True))

            if to_break:
                break

            #print("GENERATED:", self.tokenizer.decode(new_token_ids, skip_special_tokens=True))

            if gamma > 0:
                for j in range(len(all_token_ids) - added_size - gamma, len(all_token_ids) - gamma):
                    #print(j, j+gamma, len(all_token_ids))
                    token_group = tuple(all_token_ids[j:j + gamma])
                    token_hash = hash(token_group)
                    start_pos = j
                    if token_hash not in self.copy_dict:
                        self.copy_dict[token_hash] = []
                    if start_pos not in self.copy_dict[token_hash]:
                        self.copy_dict[token_hash].append(start_pos)

            gamma = stored_gamma

        #print("!!!", (len(all_token_ids)-prompt_length))
        print("!!!", max_new_tokens)

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

    def generate(self, prompt, temperature=0.0, top_k=0, top_p=1.0, k=10, gamma=5, max_new_tokens=100):

        all_token_ids, _ = self.generate_raw(prompt, temperature, top_k, top_p, k, gamma, max_new_tokens=max_new_tokens)

        return self.tokenizer.decode(all_token_ids[0]), self.total_accepted


    def target_generate_greedy_temp(self, prompt, max_new_tokens=50):
        """
        Generate text one token at a time using greedy decoding with KV caching.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        import torch
        
        # Tokenize the input prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Initial forward pass to get logits and past_key_values
        # `use_cache=True` ensures that past_key_values are returned
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
        print("GENERATING DEFAULT")
        tokenss = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        greedy_output = tokenss.squeeze(0).tolist()
        past_key_values = None
        stop_token = self.tokenizer.eos_token_id

        # We have to generate the tokens one by one so that we can use the same sampling function to guarantee the same outputs
        # This doesn't significantly affect performance.
        for _ in range(max_new_tokens):
            outputs = self.target_model(
                tokenss,
                return_dict=True,
                past_key_values=past_key_values,
                use_cache=True
            )
            target_logits = outputs.logits
            target_probs = self.sample(target_logits, temperature, top_k, top_p)

            next_token = torch.multinomial(target_probs[:, -1], 1)
            greedy_output.append(next_token.item())
            if greedy_output[-1] == stop_token:
                break

            tokenss = torch.tensor([[greedy_output[-1]]], device=self.device)
            past_key_values = outputs.past_key_values

        greedy_output = torch.tensor([greedy_output], device=self.device)
        return greedy_output

    def target_generate_greedy(self, prompt, max_new_tokens=50):
        return self.tokenizer.decode(self.target_generate_greedy_raw(prompt, max_new_tokens)[0])

    #def draft_generate_greedy(self, prompt, max_new_tokens=50):
    #    # Greedy decoding with the draft model
    #    model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    #    greedy_output = self.draft_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    #    return self.tokenizer.decode(greedy_output[0])







    def generate_raw_regenerateKV(self, prompt, temperature=0.0, top_k=0, top_p=1.0, k=10, gamma=5, max_new_tokens=100):

        self.copy_dict = dict()
        use_specdec = False
        stored_gamma = gamma
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        self.preprocess_prompt(prompt_ids, gamma)
        all_token_ids = prompt_ids.squeeze(0).tolist()
        prompt_length = len(all_token_ids)
        total_generated = 0
        target_past_key_values = None
        draft_past_key_values = None

        with torch.no_grad():
            target_outputs = self.target_model(prompt_ids, use_cache=True, return_dict=True)
        target_past_key_values = target_outputs.past_key_values

        target_logits = target_outputs.logits
        target_probs = self.sample(target_logits, temperature, top_k, top_p)
        next_token = torch.multinomial(target_probs[:, -1], 1)
        new_token_ids = next_token.squeeze(0).tolist()
        if not isinstance(new_token_ids, list):
            new_token_ids = [new_token_ids]
        all_token_ids.extend(new_token_ids)
        total_generated += len(new_token_ids)

        self.total_accepted = 0

        stop_token = self.tokenizer.eos_token_id

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
                    to_add = min(10, left_tokens)

                    draft_chunk = all_token_ids[(first_occurrence + gamma): (first_occurrence + gamma + to_add)]
                    draft_tokens = torch.tensor(draft_chunk, device=self.device).unsqueeze(0)
                    last_token_id = all_token_ids[-1]
                    last_token_tensor = torch.tensor([[last_token_id]], dtype=draft_tokens.dtype, device=draft_tokens.device)
                    draft_tokens = torch.cat([last_token_tensor, draft_tokens], dim=1)
                    
                    vocab_size = self.vocab_size
                    draft_probs = torch.zeros((draft_tokens.size(1), 1, vocab_size), device=draft_tokens.device)
                    draft_probs[torch.arange(draft_tokens.size(1)), 0, draft_tokens] = 1
                    #gamma = draft_tokens.size(1)-1
                    draft_size = draft_tokens.size(1)-1

            #copied = False

            if draft_tokens is not None:

                with torch.no_grad():
                    target_outputs = self.target_model(draft_tokens, use_cache=True, return_dict=True, past_key_values=target_past_key_values)
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
                self.summed_query += len(accepted_tokens)/draft_tokens.size(1)
                self.total_query += 1

                num_accepted = len(accepted_tokens)
                self.total_accepted += num_accepted-1
                
                accepted_tokens_tensor = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
                accepted_tokens_tensor = torch.cat([last_token_tensor, accepted_tokens_tensor], dim=1)

                with torch.no_grad():
                    target_outputs = self.target_model(
                        accepted_tokens_tensor,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=target_past_key_values,
                    )
                target_past_key_values = target_outputs.past_key_values
                #copied = True

            else:
                #num_accepted = 1
                last_token_id = all_token_ids[-1]
                input_for_target = torch.tensor([[last_token_id]], device=self.device)
                with torch.no_grad():
                    target_outputs = self.target_model(input_for_target, use_cache=True, return_dict=True, past_key_values=target_past_key_values)
                target_past_key_values = target_outputs.past_key_values
            target_logits = target_outputs.logits
            target_probs = self.sample(target_logits, temperature, top_k, top_p)
            next_token = torch.multinomial(target_probs[:, -1], 1)
            if new_tokens is None:
                new_tokens = next_token
            else:
                new_tokens = torch.cat([new_tokens.unsqueeze(0), next_token], dim=1)

            to_break = False

            new_token_ids = new_tokens.squeeze(0).tolist()
             
            if not isinstance(new_token_ids, list):
                new_token_ids = [new_token_ids]

            if stop_token in new_token_ids:
                stop_index = new_token_ids.index(stop_token)  
                new_token_ids = new_token_ids[:stop_index]
                to_break = True
            
            all_token_ids.extend(new_token_ids)
            total_generated += len(new_token_ids)
            added_size = len(new_token_ids)

            if to_break:
                break

            if gamma > 0:
                for j in range(len(all_token_ids) - added_size - gamma, len(all_token_ids) - gamma):
                    #print(j, j+gamma, len(all_token_ids))
                    token_group = tuple(all_token_ids[j:j + gamma])
                    token_hash = hash(token_group)
                    start_pos = j
                    if token_hash not in self.copy_dict:
                        self.copy_dict[token_hash] = []
                    if start_pos not in self.copy_dict[token_hash]:
                        self.copy_dict[token_hash].append(start_pos)

            gamma = stored_gamma

            #if copied:
            #   print(self.tokenizer.decode(all_token_ids))

        self.total_generated += len(all_token_ids) - prompt_length
        self.summed_copied += self.total_accepted

        if len(all_token_ids) > max_new_tokens+prompt_length:
            all_token_ids = all_token_ids[0:(max_new_tokens+prompt_length)]

        print("So far we accepted", (self.summed_copied/self.total_generated*100), "out of each 100 tokens")
        print("We attempted to copy", self.total_query, "times")
        print("Out of those we accepted ", (self.summed_query/self.total_query*100), "tokens for each 100 tokens")

        #print("FINAL ACCEPTED", self.tokenizer.decode(all_token_ids))

        all_token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long)
        all_token_ids_tensor = all_token_ids_tensor.unsqueeze(0)

        return all_token_ids_tensor, self.total_accepted