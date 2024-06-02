###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-02 11:51:34
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-02 11:59:52
 # @FilePath: /llm-bias/running_scripts/mit_server_g0.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

CUDA_VISIBLE_DEVICES=0 python generation_benchmark --model_name_or_path=meta-llama/Llama-2-7b-chat-hf --datasets=bookcorpus_new.jsonl --bias_types=lgbt, appearance, class, education, disability, national