import argparse
import os

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_clm import evaluate
from run_clm import gen_completions

# data_list = [
#     "bold.jsonl", "imdb_new.jsonl", "realtoxic_new.jsonl", "jigsaw_new.jsonl",
#     "wikitext_new.jsonl", "wikitoxic_new.jsonl", "cnn_dailymail_new.jsonl"
# ]
data_list = ["wikitext_new.jsonl"]


def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        nargs='+',
        default=["Salesforce/ctrl"],
        # default=[
        #     'openai-community/gpt2', 'xlnet/xlnet-base-cased',
        #     'meta-llama/Llama-2-7b-chat-hf', 'facebook/opt-1.3b',
        #     'Salesforce/ctrl'
        # ],
        help=
        'The name or path of the model(s) to benchmark (provide multiple models separated by space)'
    )
    parser.add_argument('--bias_types',
                        type=str,
                        required=False,
                        nargs='+',
                        default=['gender', 'race', 'age', 'occupation', 'religion'],
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
                df = df.reset_index(drop=True)
                model = AutoModelForCausalLM.from_pretrained(model_)
                tokenizer = AutoTokenizer.from_pretrained(model_)
                # tokenizer.pad_token = tokenizer.eos_token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.padding_side = 'left'
                output_filename = f"{model_}_{data[:data.find('.json')]}_normal.json"
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
                completions_filename = f"outputs/completions/{model_}_{data[:data.find('.json')]}_normal.json"
                df = pd.read_json(completions_filename, lines=True)
                df = df.reset_index(drop=True)
                output_log = f"eval_output_{data[:data.find('.json')]}_normal.log"
                evaluate(df, output_log)


if __name__ == '__main__':
    main()
