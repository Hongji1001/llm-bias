###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-05 11:22:05
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-06 04:29:26
 # @FilePath: /llm-bias/running_scripts/generate_metric_data.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
# export OMP_NUM_THREADS=80
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

model_path=/data1/TxPLM/llm_ckpt/meta-llama/Llama-2-7b-chat-hf

CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_bookcorpus_new_UNLEARN --datasets cnn_dailymail_new.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_bookcorpus_new_UNLEARN --datasets bold.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_bookcorpus_new_UNLEARN --datasets bookcorpus_new.jsonl

CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_cnn_dailymail_new_UNLEARN  --datasets cnn_dailymail_new.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_cnn_dailymail_new_UNLEARN  --datasets bold.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_cnn_dailymail_new_UNLEARN  --datasets bookcorpus_new.jsonl

CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_unlearned  --datasets cnn_dailymail_new.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_unlearned  --datasets bold.jsonl
CUDA_VISIBLE_DEVICES=0 python generation_benchmark_no_metric.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/opt_unlearned  --datasets bookcorpus_new.jsonl