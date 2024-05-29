import argparse
import logging
import os

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import get_newest_folder
import train_clm
from dataset import CLMFineTuningDataset
from eval_clm import evaluate
from models import INLPGPT2LMHeadModel
from run_clm import gen_completions
from src.CDA import swap_gender_terms

DATA_LIST = [
    "bold.jsonl", "imdb_2k.jsonl", "realtoxic_2k.jsonl", "jigsaw_2k.jsonl",
    "wikitext_2k.jsonl", "wikitoxic_2k.jsonl", "cnn_dailymail_2k.jsonl"
]

models = ["GPT2", "XLNet", "Llama2", "Opt1_3", "CTRL"]

# CDA
# finetuning
def CDA_finetuning(bias_type, models):
    # NOTE: CDA only support gender right now
    print("CDA finetuning")
    print()

    for model_name, model_path in models.items():
        for dataset in DATA_LIST:
            with open(f'words/gender.yaml', 'r', encoding='utf-8') as yaml_file:
                gender = yaml.safe_load(yaml_file)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            df = pd.read_json(f"data/{dataset}")
            df = df[df['domain'] == bias_type]
            df = swap_gender_terms(df, gender)
            df = df.reset_index(drop=True)
            train_dataset = CLMFineTuningDataset(df['texts'], tokenizer)
            output_filename = f"checkpoint/{model_name}_{dataset[:dataset.find('.json')]}_CDA"
            train_clm.train_clm(model, tokenizer, train_dataset,
                                output_filename)


# running
def CDA_generating(bias_type, models):
    print("CDA generating")
    print()

    for model_name, model_path in models.items():
        for data in DATA_LIST:
            df = pd.read_json(f"data/{data}")
            df = df[df['domain'] == bias_type]
            df = df.reset_index(drop=True)
            model_name_or_path = f"checkpoint/{model_name}_{data[:data.find('.json')]}_CDA"
            model = AutoModelForCausalLM.from_pretrained(
                get_newest_folder(model_name_or_path))
            tokenizer = AutoTokenizer.from_pretrained(
                get_newest_folder(model_name_or_path))
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            output_filename = f"{model_name}_{data[:data.find('.json')]}_CDA.json"
            gen_completions(model, tokenizer, df, output_filename)


# evaluating
def CDA_eval(models):
    print("CDA evaluating")
    print()

    for model_name, model_path in models.items():
        for data in DATA_LIST:
            completions_filename = f"outputs/completions/{model_name}_{data[:data.find('.json')]}_CDA.json"
            df = pd.read_json(completions_filename, lines=True)
            df = df.reset_index(drop=True)
            output_log = f"eval_output_{data[:data.find('.json')]}_CDA.log"
            evaluate(df, output_log)


# INLP
def INLP_projecting(bias_type, models):
    for model_name, model_path in models.items():
        os.system(
            'conda run -n bias-bench ' +
            'python bias-bench/experiments/inlp_projection_matrix.py ' +
            f'--model {model_name}Model --model_name_or_path {model_path} --bias_type {bias_type}'
        )


# running
def INLP_generating(bias_type, models):
    print("INLP generating")
    print()

    for model_name, model_path in models.items():
        for data in DATA_LIST:
            df = pd.read_json(f"data/{data}")
            df = df[df['domain'] == bias_type]
            df = df.reset_index(drop=True)

            projection_matrix_path = f"bias-bench/bias-bench/results/projection_matrix/projection_m-{model_name}Model_c-{model_path}_t-{bias_type}_s-0.pt"
            projection_matrix = torch.load(projection_matrix_path)

            # Dynamically load the model class
            model_class_name = f"INLP{model_name.replace('-', '').replace('_', '')}LMHeadModel"
            try:
                model_class = getattr(__import__('models'), model_class_name)
            except AttributeError:
                logging.error(f"Model class {model_class_name} not found in my_project module.")
                continue

            model = model_class(model_name, projection_matrix)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

            output_filename = f"{model_name}_{data[:data.find('.json')]}_INLP.json"
            gen_completions(model, tokenizer, df, output_filename)


# evaluating
def INLP_eval(models):
    print("INLP evaluating")
    print()
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            completions_filename = f"outputs/completions/{model_name}_{data[:data.find('.json')]}_INLP.json"
            df = pd.read_json(completions_filename, lines=True)
            df = df.reset_index(drop=True)
            output_log = f"eval_output_{data[:data.find('.json')]}_INLP.log"
            evaluate(df, output_log)


# self debias
# running
def SELFDEBIAS_generating(bias_type, models):
    print("self debias only support gpt2 now")
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            data_raw = pd.read_json(f"data/{data}", lines=True)
            data_raw = data_raw[data_raw['domain'] == bias_type]
            data_raw.to_json(f'data/temp_{data}', orient='records', lines=True)
            os.system(
                "conda run -n bias-bench " +
                "python bias-bench/bias_bench/debias/self_debias/self_debiasing.py "
                +
                f"--prompts_filename data/temp_{data} --models {model_name} " +
                "--output_dir outputs/completions/ --api_key AIzaSyCH1mAOwrUSdt4j8FWyaTiq7zZRuNZTuw8  --modes debiased --max_length 128"
            )
            os.system(f"rm data/temp_{data}")
    pass


# evaluating
def SELFDEBIAS_eval(models):
    print("SELFDEBIAS evaluating")
    print()
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            completions_filename = f"outputs/completions/{model_name}_{data[:data.find('.json')]}_selfbias.jsonl"
            df = pd.read_json(completions_filename, lines=True)
            df = df.reset_index(drop=True)
            output_log = f"eval_output_{data[:data.find('.json')]}_SELFDEBIAS.log"
            evaluate(df, output_log)


# unlearning
# finetuning
def UNLEARNING_finetuning(bias_type, models):
    print("UNLEARNING finetuning")
    print()
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            data_raw = pd.read_json(f"data/{data}", lines=True)
            data_raw = data_raw[data_raw['domain'] == bias_type]
            data_raw.to_json(f'data/temp_{data}', orient='records', lines=True)
            os.system(
                "conda run -n llm-unlearning " +
                "python llm_unlearn/unlearn_harm.py " +
                f"--model_name {model_name} --model_save_dir checkpoint/{model_name}_{data[:data.find('.jsonl')]}_UNLEARN --datapath data/temp_{data}"
            )
            os.system(f"rm data/temp_{data}")
    pass


# running
def UNLEARNING_generating(bias_type, models):
    print("UNLEARNING generating")
    print()
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            df = pd.read_json(f"data/{data}")
            df = df[df['domain'] == bias_type]
            df = df.reset_index(drop=True)
            model_name_or_path = f"checkpoint/{model_name}_{data[:data.find('.jsonl')]}_UNLEARN"
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            output_filename = f"{model_name}_{data[:data.find('.json')]}_unlearn.jsonl"
            gen_completions(model, tokenizer, df, output_filename)


# evaluating
def UNLEARNING_eval(models):
    print("UNLEARNING evaluating")
    print()
    for model_name, model_path in models.items():
        for data in DATA_LIST:
            completions_filename = f"outputs/completions/{model_name}_{data[:data.find('.json')]}_unlearn.jsonl"
            df = pd.read_json(completions_filename, lines=True)
            df = df.reset_index(drop=True)
            output_log = f"eval_output_{data[:data.find('.json')]}_UNLEARN.log"
            evaluate(df, output_log)


def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument('--method',
                        type=str,
                        required=True,
                        nargs='+',
                        help='The bias mitigation methods to reproduce')
    parser.add_argument('--bias_type',
                        type=str,
                        required=True,
                        help='The type of bias to benchmark')
    args = parser.parse_args()
    models = {"GPT2": "openai-community/gpt2", "XLNet": "xlnet/xlnet-base-cased", "Llama2": "meta-llama/Llama-2-7b-chat-hf", "Opt1_3": "facebook/opt-1.3b", "CTRL": "Salesforce/ctrl"}
    for expr in args.method:
        if args.method == "CDA":
            CDA_finetuning(args.bias_type, models)
            CDA_generating(args.bias_type, models)
            CDA_eval(models)
        if args.method == "INLP":
            INLP_projecting(args.bias_type, models)
            INLP_generating(args.bias_type, models)
            INLP_eval(models)
        if args.method == "SELFDEBIAS":
            SELFDEBIAS_generating(args.bias_type, models)
            SELFDEBIAS_eval(models)
        if args.method == "UNLEARN":
            UNLEARNING_finetuning(args.bias_type, models)
            UNLEARNING_generating(args.bias_type, models)
            UNLEARNING_eval(models)


if __name__ == '__main__':
    main()
