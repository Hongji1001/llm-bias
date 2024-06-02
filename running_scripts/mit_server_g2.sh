###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-02 11:51:34
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-02 11:58:48
 # @FilePath: /llm-bias/running_scripts/mit_server_g3.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

CUDA_VISIBLE_DEVICES=2 python generation_benchmark --model_name_or_path=xlnet/xlnet-base-cased --datasets=bookcorpus_new.jsonl