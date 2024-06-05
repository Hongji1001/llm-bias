
export TRANSFORMERS_CACHE="/data1/tianlong/.cache"
export HOME=/data1/cache_tlc

model_path=/data1/TxPLM/llm_ckpt/facebook/opt-1.3b

CUDA_VISIBLE_DEVICES=1 python llm-unlearning/unlearn_harm.py --model_name $model_path --model_save_dir checkpoint/opt_unlearned --log_file logs/opt-unlearn.log --use_lora
