import argparse
from pathlib import Path
import random

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import set_seed
from eval_clm import evaluate
from run_clm import gen_completions
import numpy as np
from gensim.models import KeyedVectors

data_list = [
    "bold.jsonl", "imdb_new.jsonl", "realtoxic_new.jsonl", "jigsaw_new.jsonl",
    "wikitext_new.jsonl", "wikitoxic_new.jsonl", "cnn_dailymail_new.jsonl"
]
# "bold.jsonl", "imdb_new.jsonl", "realtoxic_new.jsonl", "jigsaw_new.jsonl",
    # "wikitext_new.jsonl", "wikitoxic_new.jsonl", "cnn_dailymail_new.jsonl"
# data_list = ["wikitext_new.jsonl"]
# 'openai-community/gpt2', 'xlnet/xlnet-base-cased', 'facebook/opt-1.3b', 'google/gemma-2b'
# 'meta-llama/Llama-2-7b-chat-hf'

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    set_seed(seed_value)

def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        nargs='+',
        default=[
            'facebook/opt-1.3b',
        ],
        help=
        'The name or path of the model(s) to benchmark (provide multiple models separated by space)'
    )
    parser.add_argument('--datasets',
                        type=str,
                        required=False,
                        nargs='+',
                        default=["cnn_dailymail_new.jsonl", "bold.jsonl"],
                        help='The dataset to benchmark')
    parser.add_argument('--bias_types',
                        type=str,
                        required=False,
                        nargs='+',
                        default=['gender', 'religion', 'age', 'race', 'bodyshaming', 'socioeconomic', 'lgbt', 'appearance', 'class', 'education', 'disability', 'national'],
                        help='The type of bias to benchmark')
    args = parser.parse_args()

    set_random_seed(0)
    dir_path = Path("outputs/completions")
    dir_path.mkdir(parents=True, exist_ok=True)
    if dir_path.exists():
        print(f"Directory {dir_path} has been created successfully.")
    else:
        print(f"Failed to create the directory {dir_path}.")

    # first step: generate completions
    for model_ in args.model_name_or_path:
        for data in args.datasets:
            for bias_type in args.bias_types:
                print(f"data/{data}")
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
                    batch_size = 32
                gen_completions(model, tokenizer, df, batch_size, output_filename)

    exit()
    dir_path = Path("outputs/metrics")
    dir_path.mkdir(parents=True, exist_ok=True)
    if dir_path.exists():
        print(f"Directory {dir_path} has been created successfully.")
    else:
        print(f"Failed to create the directory {dir_path}.")

    # second step: eval completions
    words_file = '~/.cache/GoogleNews-vectors-negative300-hard-debiased.txt'
    print("loading", words_file)
    glove_model = KeyedVectors.load_word2vec_format(words_file,
                                                    binary=False,
                                                    unicode_errors='ignore')
    for model_ in args.model_name_or_path:
        for data in args.datasets:
            for bias_type in args.bias_types:
                completions_filename = f"outputs/completions/{model_.replace('/', '-')}_{bias_type}_{data[:data.find('.json')]}_normal.json"
                if not Path(completions_filename).exists():
                    print(f"File not found: {completions_filename}. Skipping...")
                    continue
                output_log = f"outputs/metrics/eval_output_{model_.replace('/', '-')}_{data[:data.find('.json')]}_{bias_type}_normal.log"
                if Path(f'{output_log}').exists():
                    print(f"File exists: outputs/metrics/{output_log}. Skipping...")
                    continue 
                df = pd.read_json(completions_filename, lines=True)
                df = df.reset_index(drop=True)
                evaluate(df, output_log, glove_model)


if __name__ == '__main__':
    main()
