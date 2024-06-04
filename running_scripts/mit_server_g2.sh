###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-02 11:51:34
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-04 17:25:20
 # @FilePath: /llm-bias/running_scripts/mit_server_g2.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

model_path=/data1/TxPLM/llm_ckpt/meta-llama/Llama-2-7b-chat-hf
# model_path=/data1/TxPLM/llm_ckpt/google/gemma-2b

# CUDA_VISIBLE_DEVICES=2 python llm_unlearn/unlearn_harm.py --model_name $model_path --model_save_dir checkpoint/llama2_unlearned --log_file logs/llama2-unlearn.log --use_lora

# CUDA_VISIBLE_DEVICES=2 python generation_benchmark.py --model_name_or_path xlnet/xlnet-base-cased --datasets bookcorpus_new.jsonl
CUDA_VISIBLE_DEVICES=2 python generation_benchmark.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/llama2_unlearned --datasets bookcorpus_new.jsonl
CUDA_VISIBLE_DEVICES=2 python generation_benchmark.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/llama2_unlearned --datasets cnn_dailymail_new.jsonl
CUDA_VISIBLE_DEVICES=2 python generation_benchmark.py --model_name_or_path /data1/TxPLM/llm-bias/checkpoint/llama2_unlearned --datasets bold.jsonl