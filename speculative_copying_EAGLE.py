import torch
import torch.nn.functional as F

from model.eagle.ea_model import *
from model.eagle.kv_cache import *
from model.eagle.utils import *
from model.eagle.choices import *

class SpeculativeDecoder:
    """
    Loads one EAGLE model and performs copy-based expansions:
      - If we can copy (k-gram) => copy chunk
      - Else => do a short tree-based decode (like official 'ea_forward') 
                for up to 'delta' tokens, accept them in full.
      - No partial acceptance per token, no token-by-token fallback.
    """

    def __init__(
        self,
        base_model_path,
        draft_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eagle_max_steps=512,
        tree_choices=None,
    ):
        """
        Args:
            base_model_path (str): Base LLaMA or similar model path
            draft_model_name (str): The EAGLE patch/delta
            device (str): 'cuda' or 'cpu'
            eagle_max_steps (int): A safety limit for tree expansions
            tree_choices (Any): The EAGLE "tree_choices" config (e.g. mc_sim_7b_63)
        """
        self.device = device
        self.eagle_max_steps = eagle_max_steps
        # Respect passed-in tree_choices or use default
        self.tree_choices = tree_choices if tree_choices is not None else [
            [0],[1],[2],[3],[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]
            ,[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
            [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,0,0],[0,0,0,0,1]
        ]

        # 1) Load the EAGLE model 
        #    (base_model_path + patch => single EaModel)
        self.vicunaa = False
        if 'vicuna' in base_model_path:
            self.vicunaa = True
        self.model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=draft_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        # Ensure model is on the correct device
        self.model.to(self.device)

        # 2) The EAGLE tokenizer
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "USER: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "ASSISTANT: {{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "ASSISTANT:"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Housekeeping for copy-based stats
        self.copy_dict = {}
        self.total_accepted = 0
        self.summed_copied = 0
        self.total_generated = 0
        self.summed_query = 0
        self.total_query = 0
        self.accepted_spec = 0
        self.total_spec = 0

        self.turn_tau = 0
        self.turn_attempted = 0

        self.dec_tau = 0
        self.dec_attempted = 0

        self.use_specdec = True

        if not self.use_specdec:
            self.dec_tau = 1
            self.dec_attempted = 1

    # ----------------------------------------------------------------
    # (A) Copy dictionary building
    # ----------------------------------------------------------------
    def preprocess_prompt(self, input_ids, k):
        """
        Precompute k-gram token hashes from the prompt for chunk-based copying.
        """
        self.copy_dict = {}
        tokens = input_ids.squeeze(0).tolist()
        for i in range(len(tokens) - k + 1):
            group = tuple(tokens[i : i + k])
            h = hash(group)
            if h not in self.copy_dict:
                self.copy_dict[h] = []
            self.copy_dict[h].append(i)

    # ----------------------------------------------------------------
    # (B) Minimal "sample" if needed
    # ----------------------------------------------------------------
    @staticmethod
    def sample(logits, temperature=0.0):
        """
        If you need a simple temperature-based approach for fallback.
        Not used by the official tree-based expansions, which do their 
        own merges. We keep it here for potential fallback usage.
        """
        if temperature <= 1e-6:
            # Argmax
            max_ids = logits.argmax(dim=-1)
            return F.one_hot(max_ids, num_classes=logits.size(-1)).float()

        return F.softmax(logits / temperature, dim=-1)

    # ----------------------------------------------------------------
    # (C) The "tree-based" decode for up to `delta` tokens
    # ----------------------------------------------------------------
    def _tree_decode_chunk(self, context_ids: torch.LongTensor, delta: int = 5):
        """
        Replicates the official 'ea_forward' steps, but only decodes
        up to `delta` new tokens from the context. 
        Accepts them as a single chunk. No partial acceptance.

        Returns the *entire* updated `input_ids`, 
        from which we can parse out the newly generated portion.
        """

        # 1) We do EXACTLY what 'ea_forward' does: reset K/V
        self.model.ea_layer.reset_kv()
        # Possibly reuse or build new tree_buffers
        if hasattr(self.model, "tree_choices") and (self.model.tree_choices == self.tree_choices):
            tree_buffers = self.model.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                self.tree_choices,
                device=self.model.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            # we also set retrieve_indices_head
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.model.base_model.lm_head.weight.device
            )
        self.model.tree_buffers = tree_buffers
        self.model.tree_choices = self.tree_choices

        # 2) Initialize or reset the model caches
        if hasattr(self.model, "past_key_values"):
            past_key_values = self.model.past_key_values
            past_key_values_data = self.model.past_key_values_data
            current_length_data = self.model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data
            ) = initialize_past_key_values(self.model.base_model)
            self.model.past_key_values = past_key_values
            self.model.past_key_values_data = past_key_values_data
            self.model.current_length_data = current_length_data
            #print("HAS KEY VALUES")

        # 3) Prepare
        input_ids = context_ids.clone()
        batch_size, cur_len = input_ids.shape
        reset_tree_mode(self.model)

        # 4) Initialize the tree with "initialize_tree"
        #    returns (tree_logits, logits, hidden_state, sample_token)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids=context_ids,
            model=self.model,
            tree_attn_mask=tree_buffers["tree_attn_mask"],
            past_key_values=self.model.past_key_values,
            logits_processor=None  # no custom processor during tree-decode
        )

        new_token = 0

        # 5) The main loop, but we only produce up to `delta` tokens
        for step_i in range(self.eagle_max_steps):
            
            #print("!", step_i, input_ids.shape[1])

            if input_ids.shape[1] >= 1948:
                break
            
            #print("TIMES", step_i)
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                None  # no custom processor
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self.model,
                tree_candidates,
                self.model.past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits,
                candidates,
                None,
                cart_candidates_prob,
                tree_logits[2],
                tree_buffers["p_indices"],
                tree_candidates,
                tree_buffers["b_indices"]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                None,
                logits,
                tree_logits,
                new_token,
                self.model.past_key_values_data,
                self.model.current_length_data,
                self.model,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            #print("HA", new_token, delta)

            if new_token >= delta:
                break

            new_segment = input_ids[0][cur_len:]
            if self.tokenizer.eos_token_id in new_segment.tolist():
                break

        return input_ids

    # ----------------------------------------------------------------
    # (D) The main generate_raw loop 
    # ----------------------------------------------------------------
    def generate_raw(
        self,
        prompt: str,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        number_copy=100,
        gamma=5,
        delta=5,
        max_new_tokens=100,
    ):

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        all_tokens = prompt_ids.squeeze(0).tolist()
        prompt_len = len(all_tokens)
        total_generated = 0
        done = False

        # Build copy dictionary
        self.preprocess_prompt(prompt_ids, gamma)

        stop_id = self.tokenizer.eos_token_id

        stop_token = self.tokenizer.eos_token_id
        processed = False

        while total_generated < max_new_tokens and not done:
            copied = False

            if len(all_tokens) >= 1948:
                break

            # Attempt speculative copy
            if len(all_tokens) >= gamma and processed:
                last_g = tuple(all_tokens[-gamma:])
                h = hash(last_g)
                if h in self.copy_dict:
                    for first_occ in self.copy_dict[h]:
                        if first_occ + gamma < len(all_tokens):
                            candidate_chunk = all_tokens[first_occ + gamma : first_occ + gamma + number_copy]
                            if not candidate_chunk:
                                continue

                            # Verify candidate_chunk with base model
                            context_ids = torch.tensor([[all_tokens[-1]] + candidate_chunk], dtype=torch.long, device=self.device)
                            verified_chunk = []

                            saved_tree_mask = self.model.base_model.model.tree_mask
                            self.model.base_model.model.tree_mask = None

                            with torch.no_grad():
                                target_outputs = self.model.base_model.model(input_ids=context_ids, use_cache=True, return_dict=True, past_key_values=self.model.past_key_values)
                            
                            self.model.base_model.model.tree_mask = saved_tree_mask

                            #target_past_key_values = target_outputs.past_key_values
                            orig = self.model.base_model.lm_head(target_outputs[0])
                            target_logits = orig#[:, -1]
                            target_probs = self.sample(target_logits, temperature)
                            #print("BBB", self.model.past_key_values)

                            broken = False
                            for i in range(context_ids.shape[1]-1):

                                draft_token = context_ids[:, i+1]
                                target_prob = target_probs[:, i, draft_token]

                                if target_prob == 1:
                                    verified_chunk.append(draft_token)
                                else:
                                    chosen_token = torch.multinomial(target_probs[:, i], 1)[0]
                                    verified_chunk.append(chosen_token.item())
                                    broken = True
                                    break

                            if not broken:
                                chosen_token = torch.multinomial(target_probs[:, -1], 1)[0]
                                verified_chunk.append(chosen_token.item())

                            num_accepted = len(verified_chunk)

                            final_length = torch.tensor(len(all_tokens) + num_accepted)


                            prev_input_len = len(all_tokens)
                            select_indices = torch.tensor([i+prev_input_len for i in range(num_accepted)])
                            past_key_values_data_list = self.model.past_key_values_data
                            current_length_data = self.model.current_length_data

                            # Update the past key values based on the selected tokens
                            # Source tensor that contains relevant past information based on the selected candidate
                            for past_key_values_data in past_key_values_data_list:
                                tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
                                # Destination tensor where the relevant past information will be stored
                                dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
                                # Copy relevant past information from the source to the destination
                                dst.copy_(tgt, non_blocking=True)

                            # Update the current length tensor (currently only support batch size is 1)
                            current_length_data.fill_(prev_input_len + tgt.shape[-2])

                            if verified_chunk:
                                all_tokens.extend(verified_chunk)
                                total_generated += len(verified_chunk)
                                self.summed_copied += len(verified_chunk)
                                copied = True
                                added_size = len(verified_chunk)

                                # Check for eos
                                if stop_id in verified_chunk:
                                    done = True

                                for j in range(len(all_tokens) - added_size - gamma, len(all_tokens) - gamma):
                                    token_group = tuple(all_tokens[j:j + gamma])
                                    token_hash = hash(token_group)
                                    start_pos = j
                                    if token_hash not in self.copy_dict:
                                        self.copy_dict[token_hash] = []
                                    if start_pos not in self.copy_dict[token_hash]:
                                        self.copy_dict[token_hash].append(start_pos)
                                    
                                break

                    if done or copied:
                        continue

            processed = True
            # If no copy, do tree decode
            context_ids = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
            new_context = self._tree_decode_chunk(context_ids, delta=delta)
            new_chunk = new_context[0][len(all_tokens):].tolist()
            all_tokens.extend(new_chunk)
            total_generated += len(new_chunk)
            self.accepted_spec += len(new_chunk)

            to_break = False

            if not isinstance(new_chunk, list):
                new_chunk = [new_chunk]

            if stop_token in new_chunk:
                stop_index = new_chunk.index(stop_token)  
                new_chunk = new_chunk[:(stop_index+1)]
                to_break = True
             
            added_size = len(new_chunk)

            # Rebuild copy dict on updated tokens
            if gamma > 0:
                for j in range(len(all_tokens) - added_size - gamma, len(all_tokens) - gamma):
                    token_group = tuple(all_tokens[j:j + gamma])
                    token_hash = hash(token_group)
                    start_pos = j
                    if token_hash not in self.copy_dict:
                        self.copy_dict[token_hash] = []
                    if start_pos not in self.copy_dict[token_hash]:
                        self.copy_dict[token_hash].append(start_pos)
                #self.preprocess_prompt(torch.tensor([all_tokens], dtype=torch.long), gamma)

            if stop_id in new_chunk:
                done = True

        # Final trim
        if len(all_tokens) > (max_new_tokens + prompt_len):
            all_tokens = all_tokens[: (max_new_tokens + prompt_len)]
        
        if self.vicunaa:
            all_tokens = all_tokens[1:-1]

        self.total_generated += (len(all_tokens) - prompt_len)

        return torch.tensor([all_tokens], dtype=torch.long, device=self.device).cpu(), self.summed_copied

    # ----------------------------------------------------------------
    # (E) The public "generate" method
    # ----------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        number_copy=100,
        gamma=5,
        delta=5,
        max_new_tokens=100
    ):
        """
        Returns (decoded_text, # tokens-copied)
        """
        token_ids, copied_count = self.generate_raw(
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            number_copy=number_copy,
            gamma=gamma,
            delta=delta,
            max_new_tokens=max_new_tokens
        )
        decoded = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

        if '<|endoftext|>' in decoded:
            decoded = decoded[:-len('<|endoftext|>')]

        if '</s>' in decoded:
            decoded = decoded[:-len(' </s>')]

        if '<s>' in decoded:
            decoded = decoded[len('<s>  '):]
        return decoded, copied_count
