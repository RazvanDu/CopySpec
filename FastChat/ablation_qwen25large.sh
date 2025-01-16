#!/bin/bash

# Define the range for gamma values
for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-72B-Instruct --model-id Qwen/Qwen25-72B-Chat-experiments-large

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-72B-Instruct --model-id Qwen/Qwen25-72B-Chat-experiments-large-copy --use-copy True --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-72B-Instruct --model-id Qwen/Qwen25-72B-Chat-large-redundant --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-72B-Instruct --model-id Qwen/Qwen25-72B-Chat-large-redundant-copy --use-copy True --gamma 3 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
done