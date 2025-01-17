#!/bin/bash


#for run in {1..2}; do
#  echo "Running HF, run=$run"
#  start_time=$(date +%s)
  
#  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-$run --delta 3 --gamma 10000

#  end_time=$(date +%s)
#  execution_time=$((end_time - start_time))
#  echo "Execution time for specdec delta=5 HF, run=$run: ${execution_time}s"
#done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-copy-3-$run --delta 3 --use-copy True --gamma 5

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for specdec delta=5 copy gamma=3, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-copy-3-$run --delta 5 --use-copy True --gamma 5

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for specdec delta=5 copy gamma=3, run=$run: ${execution_time}s"
done

#for run in {1..2}; do
#  echo "Running RED HF, run=$run"
#  start_time=$(date +%s)
  
#  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-redundant-$run --delta 3 --use-redundant True --gamma 10000

#  end_time=$(date +%s)
#  execution_time=$((end_time - start_time))
#  echo "Execution time for RED specdec delta=5 HF, run=$run: ${execution_time}s"
#done

for run in {1..2}; do
  echo "Running RED SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-copy-3-redundant-$run --delta 3 --use-copy True --gamma 5 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED specdec delta=5 copy gamma=3, run=$run: ${execution_time}s"
done


for run in {1..2}; do
  echo "Running RED SPEC, run=$run"
  start_time=$(date +%s)
  
  python fastchat/llm_judge/gen_model_answer.py --model-path Qwen/Qwen2.5-32B-Instruct --draft-path Qwen/Qwen2.5-7B-Instruct --model-id Qwen/Qwen2.5-32B-Chat-experiments-large-specdec-5-copy-3-redundant-$run --delta 5 --use-copy True --gamma 5 --use-redundant True

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for RED specdec delta=5 copy gamma=3, run=$run: ${execution_time}s"
done
