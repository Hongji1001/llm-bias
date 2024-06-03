###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-02 11:51:34
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-03 15:19:26
 # @FilePath: /llm-bias/running_scripts/mit_server_g2.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

# model_path=/data1/TxPLM/llm_ckpt/meta-llama/Llama-2-7b-chat-hf
model_path=/data1/TxPLM/llm_ckpt/google/gemma-2b

CUDA_VISIBLE_DEVICES=2 python llm_unlearn/unlearn_harm.py --model_name $model_path --model_save_dir models/llama2_unlearned --log_file logs/llama2-unlearn.log

# CUDA_VISIBLE_DEVICES=2 python generation_benchmark.py --model_name_or_path xlnet/xlnet-base-cased --datasets bookcorpus_new.jsonl