#! /bin/bash
eval "$(conda shell.bash hook)"
conda env create -f env.yaml -n biabenchmark
conda activate biabenchmark

export OMP_NUM_THREADS=60
python generation_dataset.py --dataset=cnn_dailymail
python generation_dataset.py --dataset=bookcorpus

export CUDA_VISIBLE_DEVICES=0
python generation_benchmark --model_name_or_path=openai-community/gpt2 --datasets=bold.jsonl &
export CUDA_VISIBLE_DEVICES=1
python generation_benchmark --model_name_or_path=meta-llama/Llama-2-7b-chat-hf --datasets=bold.jsonl &
export CUDA_VISIBLE_DEVICES=2
python generation_benchmark --model_name_or_path=xlnet/xlnet-base-cased --datasets=bold.jsonl &
export CUDA_VISIBLE_DEVICES=3
python generation_benchmark --model_name_or_path=facebook/opt-1.3b --datasets=bold.jsonl &

export CUDA_VISIBLE_DEVICES=0
python generation_benchmark --model_name_or_path=openai-community/gpt2 --datasets=cnn_dailymail_new &
export CUDA_VISIBLE_DEVICES=1
python generation_benchmark --model_name_or_path=meta-llama/Llama-2-7b-chat-hf --datasets=cnn_dailymail_new &
export CUDA_VISIBLE_DEVICES=2
python generation_benchmark --model_name_or_path=xlnet/xlnet-base-cased --datasets=cnn_dailymail_new &
export CUDA_VISIBLE_DEVICES=3
python generation_benchmark --model_name_or_path=facebook/opt-1.3b --datasets=cnn_dailymail_new &

export CUDA_VISIBLE_DEVICES=0
python generation_benchmark --model_name_or_path=openai-community/gpt2 --datasets=bookcorpus_new.jsonl &
export CUDA_VISIBLE_DEVICES=1
python generation_benchmark --model_name_or_path=meta-llama/Llama-2-7b-chat-hf --datasets=bookcorpus_new.jsonl &
export CUDA_VISIBLE_DEVICES=2
python generation_benchmark --model_name_or_path=xlnet/xlnet-base-cased --datasets=bookcorpus_new.jsonl &
export CUDA_VISIBLE_DEVICES=3
python generation_benchmark --model_name_or_path=facebook/opt-1.3b --datasets=bookcorpus_new.jsonl &
