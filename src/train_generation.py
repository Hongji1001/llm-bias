import argparse
import json
import os
import time

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GenerationDataset, data_loader
from metrics import guard, honest, regard, toxicity
from model import load_model_Generation

batch_size = 16 if torch.cuda.is_available() else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataset(dataset, status):
    # Load raw data from the dataset
    data_raw = data_loader(dataset)
    # Ensure the data is a Pandas DataFrame
    assert isinstance(
        data_raw,
        pd.DataFrame), "The variable is not a Pandas DataFrame data type"
    data_raw = data_raw if torch.cuda.is_available() else data_raw.head(100)

    # Split the data
    train_data, val_data = train_test_split(data_raw,
                                            test_size=0.2,
                                            stratify=data_raw['sensitive'])
    if status == "train":
        # Reset index for data consistency
        texts = train_data['prompts'].reset_index(drop=True)
        labels = train_data['texts'].reset_index(drop=True)
        sensitives = None
    else:
        texts = val_data['prompts'].reset_index(drop=True)
        labels = val_data['texts'].reset_index(drop=True)
        sensitives = val_data['sensitive'].reset_index(drop=True)

    return texts, labels, sensitives


def construct_model_path(model_name, dataset):
    """
    Constructs a file path for saving or loading a model. The path is in the
    'checkpoint' directory, located in the parent directory of the current
    file's directory. The final path includes the model name and dataset.

    Args:
    - model_name (str): The name of the model.
    - dataset (str): The name of the dataset.

    Returns:
    - str: The constructed file path.
    """

    # Get the directory of the current script
    current_file_directory = os.path.dirname(__file__)

    # Get the parent directory of the current script
    parent_directory = os.path.dirname(current_file_directory)

    # Construct the path to the 'checkpoint' directory in the parent directory
    checkpoint_directory = os.path.join(parent_directory, 'checkpoint')

    # Ensure the 'checkpoint' directory exists, create if it does not
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    # Construct the final path with model_name and dataset
    final_path = os.path.join(checkpoint_directory, f"{model_name}_{dataset}")

    return final_path


def train(model, data_loader, optimizer, device):
    model.train()
    total_loss, total_accuracy = 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, desc='Training', unit='batch'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def test(model, tokenizer, data_loader, sensitives, device):
    model.eval()
    completions, completions_split, references = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', unit='batch'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     max_length=512,
                                     num_beams=5,
                                     no_repeat_ngram_size=2,
                                     early_stopping=True)

            for output in outputs:
                completion = tokenizer.decode(output, skip_special_tokens=True)
                completions.append(completion)
                completions_split.append(completion.split())

            for label in labels:
                reference = tokenizer.decode(label, skip_special_tokens=True)
                references.append([reference.split()])

    data_to_save = {
        "completions": completions,
        "completions_split": completions_split,
        "sensitives": sensitives.to_list(),
        "references": references
    }

    output_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'outputs')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    final_path = os.path.join(
        output_file_path,
        f"{model.config.model_type}_{data_loader.dataset.dataset}.json")

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    print(f"output file save to {final_path}")

    return


def evaluate(path):
    start_time = time.time()
    with open(path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)

    completions = loaded_data["completions"]
    completions_split = loaded_data["completions_split"]
    sensitives = loaded_data["sensitives"]
    references = loaded_data["references"]

    scores = {}
    scores["bleu"] = corpus_bleu(references, completions_split)
    scores["toxicity"] = toxicity(completions, sensitives)
    scores["regard"] = regard(completions, sensitives)
    scores["honest"] = honest(completions, sensitives)
    scores["guard"] = guard(completions, sensitives)

    end_time = time.time()
    print("=" * 100)
    print("Evaluation Scores:")
    print("=" * 100)
    for metric, score in scores.items():
        print(f"{metric.capitalize()} Score: {score}")
    print("=" * 100)

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time} seconds")


def main(model_name, dataset, status, model_path=None):
    texts, labels, sensitives = prepare_dataset(dataset, status)
    if status == "train":
        model, tokenizer = load_model_Generation(status, model_name=model_name)
        model.to(device)
        train_dataset = GenerationDataset(texts, labels, tokenizer, dataset)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        epochs = 1
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device)
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f"Train Loss: {train_loss}")

        path = construct_model_path(model_name, dataset)
        model.save_pretrained(path)

    elif status == "test":
        if model_path is None:
            # Automatically construct the model path for testing if not provided
            model_path = construct_model_path(model_name, dataset)
            print(f"Constructed model path for testing: {model_path}")
        model, tokenizer = load_model_Generation(status, model_path=model_path)
        model.to(device)
        val_dataset = GenerationDataset(texts, labels, tokenizer, dataset)
        val_loader = DataLoader(val_dataset, batch_size)
        test(model, tokenizer, val_loader, sensitives, device)

    else:
        evaluate(
            os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'outputs'), f"{model_name}_{dataset}.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')

    # Add an argument for the model, this is a required argument
    parser.add_argument('--model',
                        type=str,
                        required=False,
                        help='The name of the model to use')

    # Add an argument for the dataset, this is also a required argument
    parser.add_argument('--dataset',
                        type=str,
                        required=False,
                        help='The name of the dataset to use')

    # Add an argument for the status, this is also a required argument
    parser.add_argument('--status',
                        type=str,
                        required=False,
                        help='The status of llm')

    # Add an argument for the name of pre-trained model, this is also a required argument
    parser.add_argument('--path',
                        type=str,
                        required=False,
                        help='The path of model ckpt')

    # Parse the arguments
    args = parser.parse_args()

    main("gpt2", "bold", "eval", args.path)
