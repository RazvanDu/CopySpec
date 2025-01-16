"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import numpy as np

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

import sys

# Add your custom path
custom_path = "/mnt/razvandu/speculative_decoding/"
if custom_path not in sys.path:
    sys.path.append(custom_path)

from speculative_copying import SpeculativeDecoder

def run_eval(
    model_path,
    draft_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    gamma,
    delta,
    oneshot,
    max_gpu_memory,
    use_copy,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                draft_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                use_copy,
                gamma,
                delta,
                oneshot,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    draft_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    use_copy,
    gamma,
    delta,
    oneshot,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    top_p = 1
    top_k = 0

    print("AAA")

    import fastchat
    print(fastchat.__file__)

    total_tokens = 0
    
    #if use_copy:


    decoder = SpeculativeDecoder(model_path, draft_model_name=draft_path)
    tokenizer = decoder.tokenizer

    

    # Start the timer
    start_time = time.time()

    #else:
    #    model, tokenizer = load_model(
    #        model_path,
    #        revision=revision,
    #        device="cuda",
    #        num_gpus=num_gpus_per_model,
    #        max_gpu_memory=max_gpu_memory,
    #        dtype=dtype,
    #        load_8bit=False,
    #        cpu_offloading=False,
    #        debug=False,
    #        
    #    )

    #decoder = SpeculativeDecoder(main_model_name, draft_model_name)

    time_taken = dict()
    generated_dict = dict()
    torch.manual_seed(123)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.cuda.manual_seed(123)  # For CUDA operations
    #torch.cuda.manual_seed_all(123)  # If using multiple GPUs
    #import random
    #random.seed(123)  # Python random library
    #import numpy as np
    #np.random.seed(123)  # NumPy

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            conv = get_conversation_template(model_id, oneshot)
            turns = []
            for j in range(len(question["turns"])):

                qs = question["turns"][j]

                categoryy = question['category'] + "_" + str(j)

                #print("TTT", )

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                #prompt = '''from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n'''
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                #print("")
                #print("INPUT:", prompt)

                # some models may error out when generating long outputs
                #try:

                input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(
                    "cuda"
                )

                start_time2 = time.time()

                if use_copy or draft_path != "":
                    output_ids, tokens_accepted = decoder.generate_raw(
                        prompt,
                        temperature=0.0,
                        top_k=top_k,
                        top_p=top_p,
                        gamma=gamma,
                        delta=delta,
                        max_new_tokens=max_new_token
                    )
                    #print("")
                    print(tokens_accepted, "tokens were accepted!")
                    #print("")
                else:
                    output_ids = decoder.target_generate_greedy_raw(
                        prompt,
                        max_new_tokens=max_new_token,
                    )
                    #output_ids = model.generate(
                    #    torch.as_tensor(input_ids).cuda(),
                    #    do_sample=do_sample,
                    #    temperature=temperature,
                    #    max_new_tokens=max_new_token,
                    #)
                
                if categoryy not in time_taken:
                    time_taken[categoryy] = 0
                    generated_dict[categoryy] = 0
                
                print("OUTPUT:", tokenizer.decode(output_ids[0], skip_special_tokens=True))
                output_ids_temp = output_ids[:, input_tokens.size(-1) :]

                total_tokens += len(output_ids_temp[0])

                time_taken[categoryy] += time.time() - start_time2
                generated_dict[categoryy] += len(output_ids_temp[0])


                
                print(total_tokens, "tokens generated thus far!")
                print("TIME:", time_taken)
                print("TOKENS:", generated_dict)

                result_dict = {key: generated_dict[key] / time_taken[key] for key in generated_dict}
                print("RESULT:", result_dict)

                #print("!!!ASDASD")
                #print("!!!!!!!!!", output)
                
                if decoder.target_config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )

                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                #except RuntimeError as e:
                #    print("THE ERROR:", e)
                #    print("ERROR question ID: ", question["question_id"])
                #    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

                #print("@@@", j, output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    # Stop the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution time is exactly: {execution_time:.2f} seconds")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
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
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
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
        "--use-redundant",
        type=bool,
        default=False,
        help="Whether to use the redundant version.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
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
    parser.add_argument(
        "--oneshot",
        type=bool,
        default=False,
        help="Whether or not to use One Shot.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    if args.use_redundant:
        question_file = f"fastchat/llm_judge/data/{args.bench_name}/question_redundant.jsonl"
    else:
        question_file = f"fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        draft_path=args.draft_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        use_copy=args.use_copy,
        gamma=args.gamma,
        delta=args.delta,
        oneshot=args.oneshot,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
