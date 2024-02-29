import argparse

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

batch_size = 32 if torch.cuda.is_available() else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def evaluate(model, tokenizer, data_loader, sensitives, device):
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

    scores = {}
    scores["bleu"] = corpus_bleu(references, completions_split)
    scores["toxicity"] = toxicity(completions, sensitives)
    scores["regard"] = regard(completions, sensitives)
    scores["honest"] = honest(completions, sensitives)
    scores["guard"] = guard(completions, sensitives)

    print("=" * 100)
    print("Evaluation Scores:")
    print("=" * 100)
    for metric, score in scores.items():
        print(f"{metric.capitalize()} Score: {score}")
    print("=" * 100)

    return


def main(model_name, dataset, status, name="bert"):
    data_raw = data_loader(dataset)
    assert isinstance(data_raw, pd.DataFrame), "变量不是 Pandas DataFrame 数据类型"
    data_raw = data_raw if torch.cuda.is_available() else data_raw.head(12)

    train_texts, val_texts, train_labels, val_labels, _, val_sensitive = train_test_split(
        data_raw['prompts'],
        data_raw['texts'],
        data_raw['sensitive'],
        test_size=0.2,
        stratify=data_raw['sensitive'])
    train_texts = train_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    val_sensitive = val_sensitive.reset_index(drop=True)
    if status == "train":
        model, tokenizer = load_model_Generation(model_name)
        model.to(device)

        train_dataset = GenerationDataset(train_texts, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_dataset = GenerationDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        epochs = 1
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device)
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f"Train Loss: {train_loss}")

        evaluate(model, tokenizer, val_loader, val_sensitive, device)

        path = '/home/hongjixu/llm-bias/' + model_name + '_' + dataset
        model.save_pretrained(path)


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
    parser.add_argument('--name',
                        type=str,
                        required=False,
                        help='The name of pre-trained model')

    # Parse the arguments
    args = parser.parse_args()

    main(args.model, args.dataset, args.status, args.name)
