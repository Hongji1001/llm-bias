###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-02 11:51:34
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-03 15:18:59
 # @FilePath: /llm-bias/running_scripts/mit_server_g0.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

model_path=/data1/TxPLM/llm_ckpt/google/gemma-2b

CUDA_VISIBLE_DEVICES=0 python llm_unlearn/unlearn_harm.py --model_name $model_path --model_save_dir checkpoint/llama2_cnn_dailymail_new_UNLEARN --datapath data/cnn_dailymail_new.jsonl

# CUDA_VISIBLE_DEVICES=0 python generation_benchmark.py --model_name_or_path $model_path --datasets bookcorpus_new.jsonl --bias_types lgbt appearance class education disability national