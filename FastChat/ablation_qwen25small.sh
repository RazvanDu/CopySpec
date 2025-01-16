#!/bin/bash

# Define the range for gamma values
for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-7B-Chat-experiments-large-$run

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-7B-Chat-experiments-large-copy-$run --use-copy True --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-7B-Chat-large-redundant-$run --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-7B-Chat-large-redundant-copy-$run --use-copy True --gamma 3 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED gamma=$gamma, run=$run: ${execution_time}s"
done