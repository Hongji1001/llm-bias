import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

from dataset import MLMClassificationDataset, load_classification_dataset


def train_mlm_for_classification(model, tokenizer, train_dataset, validation_dataset, output_filename):
    training_args = TrainingArguments(
        output_dir=output_filename,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        warmup_steps=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.dataset_file is not None:
        df = pd.read_json(args.dataset_file, lines=True)
        train_dataset = MLMClassificationDataset(df['text'], df['label'], tokenizer)
        num_labels = train_dataset['label'].nunique()
    elif args.dataset_name is not None:
        train_dataset = load_classification_dataset(args.dataset_name)
        train_dataset = train_dataset[train_dataset['split'] == 'train'].reset_index(drop=True)
        num_labels = train_dataset['label'].nunique()
        train_dataset = MLMClassificationDataset(train_dataset['text'], train_dataset['label'], tokenizer)
    else:
        raise argparse.ArgumentError('give at least one in --dataset_file and --dataset_name')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    train_mlm_for_classification(model, tokenizer, train_dataset, f'checkpoint/{args.model_name_or_path}_{args.dataset_name}')
    