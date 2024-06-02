import argparse
from util import get_newest_folder
from dataset import MLMClassificationDataset, load_classification_dataset
from train_mlm import train_mlm_for_classification
from run_mlm import test_mlm_for_classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from pathlib import Path

model_names = [
    "bert-base-uncased", "roberta-base", "albert-base-v2",
    "distilbert-base-uncased", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "microsoft/deberta-v3-base"
]
data_list = [
    "equibench"
]

# data_list = ["jigsaw"]


# training
def train():
    for model_name in model_names:
        for data in data_list:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # train_dataset = load_classification_dataset(data)
            train_dataset = pd.read_json(f'data/gen_{data}.jsonl', lines=True)
            print(train_dataset)
            train_dataset = train_dataset[train_dataset['split'] ==
                                          'train'].reset_index(drop=True)
            num_labels = train_dataset['label'].nunique()
            train_dataset = MLMClassificationDataset(train_dataset['text'],
                                                     train_dataset['label'],
                                                     tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels)
            train_mlm_for_classification(
                model, tokenizer, train_dataset,
                f'checkpoint/{model_name}_gen_{data}_finetune')


# testing
def test():
    for model_name in model_names:
        for data in data_list:
            model_name_or_path = f"checkpoint/{model_name}_{data}_finetune"
            tokenizer = AutoTokenizer.from_pretrained(
                get_newest_folder(model_name_or_path))
            train_dataset = load_classification_dataset(data)
            # train_dataset = pd.read_json(f'data/gen_{data}.jsonl', lines=True)
            train_dataset = train_dataset[train_dataset['split'] ==
                                          'test'].reset_index(drop=True)
            num_labels = train_dataset['label'].nunique()
            model = AutoModelForSequenceClassification.from_pretrained(
                get_newest_folder(model_name_or_path), num_labels=num_labels)
            output_file_path = Path(__file__).resolve().parent / 'outputs'
            output_file_path.mkdir(parents=True, exist_ok=True)
            output_file_path = output_file_path / 'metrics'
            output_file_path.mkdir(parents=True, exist_ok=True)
            filename = output_file_path / f"eval_output_{data}_{model_name}.log"
            test_mlm_for_classification(model, tokenizer, train_dataset,
                                        filename)


def main():
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        nargs='+',
        default=[
            "bert-base-uncased", "roberta-base", "albert-base-v2",
            "distilbert-base-uncased"
        ],
        help=
        'The name or path of the model(s) to benchmark (provide multiple models separated by space)'
    )
    parser.add_argument('--protected_groups',
                        type=str,
                        required=False,
                        nargs='+',
                        default=[
                            'gender', 'religion', 'age', 'race', 'bodyshaming',
                            'socioeconomic', 'lgbt', 'appearance', 'class',
                            'education', 'disability', 'national'
                        ],
                        help='Type of protected group to test')
    args = parser.parse_args()

    # first step: fine-tuning on benchmark classification dataset
    for model_ in args.model_name_or_path:
        for data in data_list:
            for protected_group in args.protected_groups:
                tokenizer = AutoTokenizer.from_pretrained(model_)
                train_dataset = load_classification_dataset(
                    data, protected_group)
                if train_dataset is None:
                    continue
                print(train_dataset)
                train_dataset = train_dataset[train_dataset['split'] ==
                                              'train'].reset_index(drop=True)
                num_labels = train_dataset['label'].nunique()
                train_dataset = MLMClassificationDataset(
                    train_dataset['text'], train_dataset['label'], tokenizer)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_, num_labels=num_labels)
                train_mlm_for_classification(
                    model, tokenizer, train_dataset,
                    f"checkpoint/{model_.replace('/', '-')}_{data}_{protected_group}_normal"
                )

    # second step: eval result of classification
    for model_ in args.model_name_or_path:
        for data in data_list:
            for protected_group in args.protected_groups:
                tokenizer = AutoTokenizer.from_pretrained(model_)
                train_dataset = load_classification_dataset(
                    data, protected_group)
                if train_dataset is None:
                    continue
                print(train_dataset)
                train_dataset = train_dataset[train_dataset['split'] ==
                                              'test'].reset_index(drop=True)
                num_labels = train_dataset['label'].nunique()
                model = AutoModelForSequenceClassification.from_pretrained(
                    get_newest_folder(
                        f"checkpoint/{model_.replace('/', '-')}_{data}_{protected_group}_normal"
                    ),
                    num_labels=num_labels)
                output_file_path = Path(__file__).resolve().parent / 'outputs'
                output_file_path.mkdir(parents=True, exist_ok=True)
                output_file_path = output_file_path / 'metrics'
                output_file_path.mkdir(parents=True, exist_ok=True)
                filename = output_file_path / f"eval_output_{data}_{protected_group}_{model_.replace('/', '-')}.log"
                test_mlm_for_classification(model, tokenizer, train_dataset,
                                            filename)


if __name__ == '__main__':
    main()
