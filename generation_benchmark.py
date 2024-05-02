import argparse
import os

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_clm import evaluate
from run_clm import gen_completions

# data_list = [
#     "bold.jsonl", "imdb_2k.jsonl", "realtoxic_2k.jsonl", "jigsaw_2k.jsonl",
#     "wikitext_2k.jsonl", "wikitoxic_2k.jsonl", "cnn_dailymail_2k.jsonl"
# ]
data_list = ["imdb_2k.jsonl"]


def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        nargs='+',
        default=[
            'openai-community/gpt2', 'xlnet/xlnet-base-cased',
            'meta-llama/Llama-2-7b-chat-hf', 'facebook/opt-1.3b',
            'Salesforce/ctrl'
        ],
        help=
        'The name or path of the model(s) to benchmark (provide multiple models separated by space)'
    )
    parser.add_argument('--bias_type',
                        type=str,
                        required=False,
                        default='gender',
                        help='The type of bias to benchmark')
    args = parser.parse_args()

    # first step: generate completions
    for model_ in args.model_name_or_path:
        for data in data_list:
            df = pd.read_json(f"data/{data}", lines=True)
            if args.bias_type is not None:
                df = df[df['domain'] == args.bias_type]
            df = df.reset_index(drop=True)
            model = AutoModelForCausalLM.from_pretrained(model_)
            tokenizer = AutoTokenizer.from_pretrained(model_)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            output_filename = f"{model_}_{data[:data.find('.json')]}_normal.json"
            gen_completions(model, tokenizer, df, output_filename)

    # second step: eval completions
    for model_ in args.model_name_or_path:
        for data in data_list:
            completions_filename = f"outputs/completions/{model_}_{data[:data.find('.json')]}_normal.json"
            df = pd.read_json(completions_filename, lines=True)
            df = df.reset_index(drop=True)
            output_log = f"eval_output_{data[:data.find('.json')]}_normal.log"
            evaluate(df, output_log)


if __name__ == '__main__':
    main()
