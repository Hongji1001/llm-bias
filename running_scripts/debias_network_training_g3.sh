
###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-06-05 12:22:00
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-05 12:39:25
 # @FilePath: /llm-bias/running_scripts/debias_network_training_g3.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 
model_path=/data1/TxPLM/llm_ckpt/facebook/opt-1.3b

CUDA_VISIBLE_DEVICES=3 python llm-unlearning/unlearn_harm.py --model_name $model_path --model_save_dir checkpoint/opt_bookcorpus_new_UNLEARN --datapath data/bookcorpus_new.jsonl --use_lora
