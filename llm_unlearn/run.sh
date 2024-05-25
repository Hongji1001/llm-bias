#! /usr/bin/bash

python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_wikitoxic_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/wikitoxic_2k.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_cnn_dailymail_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/cnn_dailymail_2k.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_bold_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/bold.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_wikitext_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/wikitext_2k.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_realtoxic_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/realtoxic_2k.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_imdb_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/imdb_2k.json
python unlearn_harm.py --model_name=gpt2 --model_save_dir=models/gpt2_jigsaw_2k_selfdebiased --log_file=logs/gpt2.log --datapath /home/hongjixu/gurobi1100/llm-bias/data/jigsaw_2k.json
