## valPlus
```
python dataset/evalplus/evalplus/evaluate.py --model "deepseek-ai/deepseek-coder-33b-instruct" --dataset humaneval --backend hf  --force-base-prompt --greedy --device_map auto --trust_remote_code true --attn_implementation flash_attention_2
```

```
python dataset/evalplus/evalplus/evaluate.py   --model "meta-llama/Llama-3.1-70B-Instruct"   --dataset humaneval   --backend spec   --force-base-prompt   --greedy   --device_map auto    --trust_remote_code true --attn_implementation flash_attention_2
```


# 🚀 Speculative Decoding

This repository contains an open-source implementation of speculative decoding for accelerating inference from large language models, as described in the paper ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) by Leviathan et al.

## 🌟 Overview

Speculative decoding is a technique that can accelerate inference from large autoregressive models like Transformers without changing the model architecture or outputs. It works by using a smaller, faster "draft" model to speculatively generate multiple tokens in parallel, which are then verified and potentially accepted by the larger "target" model.

Key benefits of speculative decoding:
- ⚡ 2-3x speedup in inference time for large language models
- 🔧 No changes required to model architecture or training
- 🎯 Identical outputs to standard decoding
- 📚 Can be applied to existing pre-trained models

This implementation provides a `SpeculativeDecoder` class that can be used with any Hugging Face transformer models to perform speculative decoding.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/speculative-decoding.git
cd speculative-decoding
pip install -r requirements.txt
```

## 🚀 Usage

The main components are:

- `speculative_decoding.py`: Contains the `SpeculativeDecoder` class implementation
- `benchmark.py`: Script to benchmark speculative decoding against standard greedy decoding

To run the benchmark:

```bash
python benchmark.py
```

This will run speculative decoding using GPT-2 Medium as the target model and DistilGPT-2 as the draft model, comparing performance to standard greedy decoding.

You can modify the models, decoding parameters, and number of tokens in the `benchmark.py` file.

## 🧠 How It Works

Speculative decoding follows these key steps:

- 📝 Use a smaller "draft" model to quickly generate multiple tokens
- 🔄 Pass these draft tokens to the larger "target" model 
- 🔍 Compare probabilities from both models to decide which draft tokens to accept
- ➕ Generate an additional token with the target model
- 🔁 Repeat the process

This allows the target model to potentially generate multiple tokens per forward pass, leading to significant speedups.

The implementation handles probability comparisons, token acceptance/rejection, and adjusted sampling when needed.

## 🛠️ Customization

You can easily use different models by changing the model names in `benchmark.py`:

```python
main_model = "gpt2-medium"  # Change to your desired target model
draft_model = "distilgpt2"  # Change to your desired draft model
```

Adjust decoding parameters like `top_k`, `top_p`, `temperature`, and `max_tokens` in the benchmark function call.

## 📜 License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

## 📚 Citation

If you use this code in your research, please cite the original paper:

```
@article{leviathan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  journal={arXiv preprint arXiv:2211.17192},
  year={2023}
}
```

## 🤝 Contributing

Contributions to improve the implementation or extend its functionality are welcome! Please feel free to submit issues or pull requests.

## 🙏 Acknowledgements

This implementation is based on the research presented in ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192). 

# SpeculativeCopying
