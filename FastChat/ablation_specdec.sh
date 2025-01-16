#!/bin/bash

# Define the range for gamma values
for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-$run

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-copy-$run --use-copy True --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-$run

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for specdec delta=5 HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-copy-$run --use-copy True --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for specdec copy gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-redundant-$run --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-copy-redundant-$run --use-copy True --gamma 3 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED HF, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-redundant-$run --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED specdec delta=5 HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running RED SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-copy-redundant-$run --use-copy True --gamma 3 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED specdec copy gamma=$gamma, run=$run: ${execution_time}s"
done
