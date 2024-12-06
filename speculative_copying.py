import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

cache_directory = "/mnt/razvandu/speculative_decoding/models_cache"

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

    def __init__(self, target_model_name, draft_model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the SpeculativeDecoder with target and draft models.

        Args:
            target_model_name (str): The name or path of the target (larger) model.
            draft_model_name (str): The name or path of the draft (smaller) model.
            device (str): The device to run the models on. Defaults to 'cuda' if available, else 'cpu'.
        """
        self.device = device
        self.target_config = config = AutoConfig.from_pretrained(target_model_name, cache_dir=cache_directory)
        self.draft_config = config = AutoConfig.from_pretrained(draft_model_name, cache_dir=cache_directory)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name, cache_dir=cache_directory).to(self.device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, cache_dir=cache_directory).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name, cache_dir=cache_directory)
        self.copy_dict = dict()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_model.eval()
        self.draft_model.eval()

        self.vocab_size = self.target_model.config.vocab_size
        #print(f"Vocabulary size: {vocab_size}")

        self.max_context_target = self.target_config.max_position_embeddings
        self.max_context_draft = self.draft_config.max_position_embeddings

        print("TEST", self.max_context_target)

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

    def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0, k=10, gamma=5, max_new_tokens=100):
        """
        Generate text using speculative decoding.

        Args:
            prompt (str): The input prompt to start generation from.
            temperature (float): The temperature for sampling. Defaults to 1.0.
            top_k (int): The number of top tokens to consider for top-k sampling. Defaults to 0 (disabled).
            top_p (float): The cumulative probability threshold for top-p sampling. Defaults to 1.0 (disabled).
            gamma (int): The number of tokens to generate speculatively in each iteration. Defaults to 5.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """

        use_specdec = False

        stored_gamma = gamma

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        self.preprocess_prompt(input_ids, gamma)

        # print(self.find_text_position("sentence that"), gamma)

        attention_mask = torch.ones_like(input_ids)
        
        while True:

            # EDIT FOR EXACTLY THE SAME SIZE
            left_to_do = max_new_tokens - (input_ids.size(1) - len(self.tokenizer.encode(prompt)))
            gamma = min(gamma, left_to_do-1)
            gamma = min(gamma, min(self.max_context_draft, self.max_context_target) - input_ids.size(1))

            #print("ZZZ", input_ids.size(1), left_to_do)

            if gamma <= 0:
                break

            draft_tokens = None

            all_token_ids = input_ids.squeeze(0).tolist()
            copied = False
            if len(all_token_ids) >= gamma:
                last_gamma_tokens = tuple(all_token_ids[-gamma:])
                token_hash = hash(last_gamma_tokens)

                if token_hash in self.copy_dict and self.copy_dict[token_hash][0]+gamma < len(all_token_ids):

                    copied = True

                    #print("FOUND A MATCH")

                    first_occurence = self.copy_dict[token_hash][0]

                    #print("RRR", first_occurence, gamma, len(all_token_ids), min(len(all_token_ids), first_occurence+100))

                    left_tokens = self.max_context_target - input_ids.size(1)
                    to_add = min(100, gamma+left_tokens)

                    draft_tokens = input_ids[:, (first_occurence+gamma):(min(len(all_token_ids), first_occurence+to_add))]

                    #print("RRR2", draft_tokens.shape, input_ids.shape)

                    # draft_tokens = draft_tokens.squeeze(0)  # Remove the batch dimension, now shape [k]

                    vocab_size = self.vocab_size

                    draft_probs = torch.zeros((draft_tokens.size(1), 1, vocab_size), device=draft_tokens.device)
                    draft_probs[torch.arange(draft_tokens.size(1)), 0, draft_tokens] = 1

                    # Verify the shape
                    #print("YYY", draft_probs.shape)  # Should be [k, 1, vocab_size]

                    gamma = draft_tokens.size(1)

            if use_specdec and not copied:

                #print("HERE??")

                gamma = k

                with torch.no_grad():
                    draft_outputs = self.draft_model.generate(
                        self.clip_context(input_ids, self.max_context_draft),
                        attention_mask=attention_mask,
                        max_new_tokens=gamma,
                        #do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                draft_tokens = draft_outputs.sequences[:, input_ids.size(1):] #torch.Size([1, 5])
                draft_probs = torch.stack(draft_outputs.scores).softmax(-1) #torch.Size([5, 1, 50257]) for GPT2

                #print("TTT", draft_tokens.shape)
                #print("UUU", draft_probs.shape)

            accepted_tokens = []

            if draft_tokens is None:
                with torch.no_grad():
                    target_outputs = self.target_model.generate(
                        self.clip_context(input_ids, self.max_context_draft),
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                        #do_sample=True,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    #target_outputs = self.target_model(
                    #    self.clip_context(input_ids, self.max_context_target),
                    #    attention_mask=attention_mask,
                    #    return_dict=True,
                    #)
                print("AYY", input_ids.shape)
                print(target_outputs.shape)
                target_logits = target_outputs[:, -1, :]
                next_token = torch.multinomial(target_logits.softmax(dim=-1), num_samples=1)
                #draft_tokens = next_token
                #input_ids = torch.cat([input_ids, draft_tokens], dim=1)
                #attention_mask = torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1)
                #return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            else:

                #print("!!!", torch.cat([input_ids, draft_tokens], dim=1).shape)

                # Target model single forward pass
                with torch.no_grad():
                    target_outputs = self.target_model(
                        self.clip_context(torch.cat([input_ids, draft_tokens], dim=1), self.max_context_target),
                        attention_mask=torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1),
                        return_dict=True,
                    )
                
                #target_logits = target_outputs.logits[:, input_ids.size(1)-1:-1]
                # THIS IS CHANGED FOR EXACT SAME OUTPUTS
                target_logits = target_outputs.logits[:, input_ids.size(1)-1:]
                target_probs = self.sample(target_logits, temperature, top_k, top_p)

                # print("AAA", target_probs)
                
                # Speculative sampling
                for i in range(gamma):
                    draft_token = draft_tokens[:, i]

                    #print("TTT", draft_token)
                    #print("WWW", torch.argmax(target_probs[:, i]))

                    draft_prob = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                    target_prob = target_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                    
                    accept_prob = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
                    #print("T", target_prob)
                    #print("D", draft_prob)
                    if torch.rand(1, device=self.device) < accept_prob:
                        accepted_tokens.append(draft_token)
                    else:
                        break
                    # if draft_prob <= target_prob:
                    #     # Accept deterministically if draft prob <= target_prob
                    #     accepted_tokens.append(draft_token)
                    # else:
                    #     # Probabilistic rejection if draft prob > target_prob
                    #     rejection_prob = 1 - target_prob / draft_prob
                    #     if torch.rand(1, device=self.device) < rejection_prob:
                    #         break
                
                num_accepted = len(accepted_tokens)

                i += num_accepted

                print(num_accepted)
            
                if num_accepted < gamma:
                    adjusted_probs = torch.clamp(target_probs[:, num_accepted] - draft_probs[num_accepted], min=0)
                    adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(adjusted_probs, num_samples=1)
                else:
                    next_token = torch.multinomial(target_probs[:, -1], num_samples=1)
            
            accepted_tokens.append(next_token)
            new_tokens = torch.cat([token.view(1, 1) for token in accepted_tokens], dim=1)
            
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens)], dim=1)

            new_token_ids = new_tokens.squeeze(0).tolist()
            all_token_ids = input_ids.squeeze(0).tolist()
            start_idx = max(0, len(all_token_ids) - len(new_token_ids) - gamma + 1)

            for j in range(start_idx, len(all_token_ids) - gamma + 1):
                token_group = tuple(all_token_ids[j:j + gamma])
                token_hash = hash(token_group)
                start_pos = j

                #print(token_group)

                if token_hash not in self.copy_dict:
                    self.copy_dict[token_hash] = []
                self.copy_dict[token_hash].append(start_pos)
            
            gamma = stored_gamma

            #print("CURRENT", self.tokenizer.decode(new_tokens[0], skip_special_tokens=True))

            #if input_ids.size(1) - len(self.tokenizer.encode(prompt)) >= max_new_tokens:
            #    break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def target_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the target model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.target_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(greedy_output[0])

    def draft_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the draft model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50

        Returns:
            str: The generated text.
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.draft_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(greedy_output[0])

