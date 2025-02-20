# CopySpec: Speculative Copying for Faster Transformer Inference

## Overview

We introduce **CopySpec**, an innovative technique designed to tackle the inefficiencies large language models (LLMs) face when generating responses that closely resemble previous outputs. **CopySpec** identifies repeated sequences in the model’s chat history and speculates that the same tokens will follow, enabling seamless copying without compromising output quality or requiring additional GPU memory.

To evaluate the effectiveness of our approach, we conducted experiments using **five LLMs** and **five datasets**: **MT-Bench, CNN/DailyMail, GSM-8K, HumanEval, and our newly created dataset, MT-Redundant**. **MT-Redundant**, introduced in this paper, transforms the second turn of MT-Bench into a request for variations of the first turn’s answer, simulating real-world scenarios where users request modifications to prior responses.

Our results demonstrate **significant speed-ups**:
- **Up to 2.35×** on CNN/DailyMail.
- **3.08×** on the second turn of select **MT-Redundant** categories.
- **2.66×** on the third turn of **GSM-8K’s self-correction tasks**.
- **49% additional speed-up** over speculative decoding on the second turn of **MT-Redundant** across all eight categories.

While LLMs, even with speculative decoding, suffer from slower inference as context sizes grow, **CopySpec leverages the expanded context to accelerate inference, making it faster as the context size increases**.

## MT-Redundant Dataset

The **MT-Redundant** dataset can be downloaded from:

[MT-Redundant Dataset](https://drive.google.com/file/d/1tpYBH7ZY7N4jKzy_kxiJcZbLX8EHQqoS/view?usp=sharing)

To use it, replace the **questions file** in MT-Bench with these new questions.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/RazvanDu/SpeculativeCopying.git
cd SpeculativeCopying
pip install -r requirements.txt
```

Set up the environment by specifying the model cache directory and the main repository path:

```bash
export cache_dir="<path-to-cache-dir>"
export copyspec_path="<path-to-copyspec-repo>"
```

For example:

```bash
export cache_dir="/mnt/razvandu/speculative_decoding/models_cache"
export copyspec_path="/mnt/razvandu/speculative_decoding/"
```

## Running Evaluation on CNN/DM

To evaluate CopySpec on the **CNN/DailyMail** dataset, navigate to the `CNNDM` directory and run:

```bash
python evaluate_cnn.py --model-path "<model-name>" --use-copy <True/False> --gamma <integer>
```

Example:

```bash
python evaluate_cnn.py --model-path "meta-llama/Llama-3.1-8B-Instruct" --use-copy True --gamma 3
```

### Parameters:
- `--model-path`: (Required) The Hugging Face model identifier.
- `--use-copy`: (Optional, default: `False`) Enables speculative copying.
- `--gamma`: (Optional, default: `3`) Sets the number of tokens searched for speculative copying.

## Running Evaluation on EvalPlus

To evaluate CopySpec on **EvalPlus**, navigate to `src/evalplus` and run:

```bash
python evalplus/evaluate.py --model "<model-name>" --dataset <dataset-name> --backend <hf/spec> --greedy --device_map "auto" --trust_remote_code true --gamma <integer>
```

Example:

```bash
python evalplus/evaluate.py --model "meta-llama/Llama-3.1-8B-Instruct" --dataset humaneval --backend spec --greedy --device_map "auto" --trust_remote_code true --gamma 3
```

### Parameters:
- `--model`: (Required) The Hugging Face model identifier.
- `--dataset`: (Required) The dataset to evaluate (e.g., `humaneval`).
- `--backend`: (Required) Can be `hf` (Hugging Face base model) or `spec` (speculative copying).
- `--gamma`: (Optional, default: `3`) Sets the number of tokens searched for speculative copying (required if `--backend spec` is used).
- `--greedy`: (Optional) Enables greedy decoding.
- `--device_map`: (Optional, default: `"auto"`) Sets device mapping for execution.
- `--trust_remote_code`: (Optional, default: `true`) Allows loading external code.

## Running MT-Bench and MT-Redundant

To evaluate CopySpec on **MT-Bench** and **MT-Redundant**, navigate to the `FastChat` directory and first install the necessary dependencies:

```bash
cd FastChat
pip install -e .
```

Then, run the evaluation command:

```bash
python fastchat/llm_judge/gen_model_answer.py --model-path "<model-name>" --model-id <model-id> --use-copy <True/False> --gamma <integer> --use-redundant <True/False>
```

Example:

```bash
python fastchat/llm_judge/gen_model_answer.py --model-path "meta-llama/Llama-3.1-8B-Instruct" --model-id "llama3-8B-experiments-redundant-copy" --use-copy True --gamma 3 --use-redundant True
```

### Parameters:
- `--model-path`: (Required) The Hugging Face model identifier.
- `--model-id`: (Required) The identifier used for experiments.
- `--use-copy`: (Optional, default: `False`) Enables speculative copying.
- `--gamma`: (Optional, default: `3`) Sets the number of tokens searched for speculative copying (required if `--use-copy True` is set).
- `--use-redundant`: (Optional, default: `False`) If set to `True`, uses the **MT-Redundant** dataset instead of **MT-Bench**.

## Running GSM

To evaluate CopySpec on the **GSM** dataset, navigate to the `GSM` directory and run:

```bash
python cp_gsm.py --model-path "<model-name>" --use-copy <True/False> --gamma <integer>
```

Example:

```bash
python cp_gsm.py --model-path "Qwen/Qwen2.5-7B-Instruct" --use-copy True --gamma 3
```

### Parameters:
- `--model-path`: (Required) The Hugging Face model identifier.
- `--use-copy`: (Optional, default: `False`) Enables speculative copying.
- `--gamma`: (Optional, default: `3`) Sets the number of tokens searched for speculative copying (required if `--use-copy True` is set).

## Citation

If you find this work useful, please cite our paper:

```
@misc{dumitru2025copyspecacceleratingllmsspeculative,
      title={CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality}, 
      author={Razvan-Gabriel Dumitru and Minglai Yang and Vikas Yadav and Mihai Surdeanu},
      year={2025},
      eprint={2502.08923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08923}, 
}
```
