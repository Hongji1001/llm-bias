import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_clm import evaluate
from run_clm import gen_completions

data_list = [
    "bold.jsonl", "imdb_new.jsonl", "realtoxic_new.jsonl", "jigsaw_new.jsonl",
    "wikitext_new.jsonl", "wikitoxic_new.jsonl", "cnn_dailymail_new.jsonl"
]
# "bold.jsonl", "imdb_new.jsonl", "realtoxic_new.jsonl", "jigsaw_new.jsonl",
    # "wikitext_new.jsonl", "wikitoxic_new.jsonl", "cnn_dailymail_new.jsonl"
# data_list = ["wikitext_new.jsonl"]
# 'openai-community/gpt2', 'xlnet/xlnet-base-cased', 'facebook/opt-1.3b', 'google/gemma-2b'
# 'meta-llama/Llama-2-7b-chat-hf'

def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        nargs='+',
        default=[
            'meta-llama/Llama-2-7b-chat-hf'
        ],
        help=
        'The name or path of the model(s) to benchmark (provide multiple models separated by space)'
    )
# 'age', 'occupation', 'profession', 'religion', 
    parser.add_argument('--bias_types',
                        type=str,
                        required=False,
                        nargs='+',
                        default=['gender', 'race', 'age', 'occupation', 'profession', 'religion', 'political_ideology'],
                        help='The type of bias to benchmark')
    args = parser.parse_args()

    # first step: generate completions
    for model_ in args.model_name_or_path:
        for data in data_list:
            for bias_type in args.bias_types:
                df = pd.read_json(f"data/{data}", lines=True)
                df = df[df['domain'] == bias_type]
                if len(df) == 0:
                    print(f"{data} do not support protected group {bias_type}")
                    continue
            
                output_filename = f"{model_.replace('/', '-')}_{bias_type}_{data[:data.find('.json')]}_normal.json"
                if Path(f'outputs/completions/{output_filename}').exists():
                    print(f"File exists: outputs/completions/{output_filename}. Skipping...")
                    continue 
                
                df = df.reset_index(drop=True)
                model = AutoModelForCausalLM.from_pretrained(model_)
                tokenizer = AutoTokenizer.from_pretrained(model_)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                if "llama" in model_:
                    batch_size = 16
                    model = model.half()
                else:
                    batch_size = 16
                gen_completions(model, tokenizer, df, batch_size, output_filename)

    # second step: eval completions
    for model_ in args.model_name_or_path:
        for data in data_list:
            for bias_type in args.bias_types:
                completions_filename = f"outputs/completions/{model_.replace('/', '-')}_{bias_type}_{data[:data.find('.json')]}_normal.json"
                if not Path(completions_filename).exists():
                    print(f"File not found: {completions_filename}. Skipping...")
                    continue
                output_log = f"eval_output_{data[:data.find('.json')]}_{bias_type}_normal.log"
                if Path(f'outputs/metrics/{output_log}').exists():
                    print(f"File exists: outputs/completions/{output_log}. Skipping...")
                    continue 
                df = pd.read_json(completions_filename, lines=True)
                df = df.reset_index(drop=True)
                evaluate(df, output_log)


if __name__ == '__main__':
    main()
