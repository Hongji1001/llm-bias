import argparse
from ctypes import ArgumentError
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from dataset import CLMFineTuningDataset
from pathlib import Path

script_dir = Path(__file__).resolve().parent


def train_clm(model, tokenizer, train_dataset, output_filename):
    training_args = TrainingArguments(
        output_dir=output_filename,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=100,
        evaluation_strategy='no',
        save_strategy='epoch',
        save_total_limit=1
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        required=True,
                        help='The name or path of the model to train')
    parser.add_argument('--tokenizer_name',
                        type=str,
                        required=True,
                        help='The name of tokenizer to load')
    parser.add_argument('--dataset_file',
                        type=str,
                        required=False,
                        help='Input training data file (a jsonl file) '
                        'should include one column name "text" as training corpus')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=False,
                        help='The name of the dataset to use (via the datasets library)')
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.dataset_file is not None:
        df = pd.read_json(args.dataset_file, lines=True)
        train_dataset = CLMFineTuningDataset(df['text'], tokenizer)
    elif args.dataset_name is not None:
         train_dataset = load_dataset(args.dataset_name)['train']
    else:
        raise ArgumentError('give at least one in --dataset_file and --dataset_name')
    train_clm(model, tokenizer, train_dataset)
