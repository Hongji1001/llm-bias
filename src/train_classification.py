import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ClassificationDataset, data_loader
from metrics import eod, kld, spd
from model import load_model_Classification, load_model_sequence_pretrain


def train(model, data_loader, optimizer, device):
    model.train()
    total_loss, total_accuracy = 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, unit='batch'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if hasattr(model, 'gpt2'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    predictions, true_labels = [], []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if hasattr(model, 'gpt2'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()

            # logits = outputs.logits
            logits = outputs if not isinstance(outputs, dict) else outputs['logits']
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(data_loader), accuracy


def test(model, data_loader, device, sensitive_list):
    model.eval()
    predictions, true_labels, target_hat_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs if not isinstance(outputs, dict) else outputs['logits']

            probabilities = torch.softmax(logits, dim=1)
            output_probs = probabilities.cpu().numpy()
            target_hat_list.append(output_probs)

            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    # print(target_hat_list)
    # print(np.array(true_labels))
    # print(sensitive_list)
    report = classification_report(true_labels, predictions)    # Adjust class names as needed
    eod(target_hat_list, np.array(true_labels), np.array(sensitive_list), 0.5)
    print("kld: ", kld(target_hat_list, np.array(sensitive_list)))
    print("spd: ", spd(target_hat_list[:, 1], np.array(sensitive_list)))
    return report


def main(model_name, dataset, status="train", name="bert"):
    data_raw = data_loader(dataset)
    assert isinstance(data_raw, pd.DataFrame), "变量不是 Pandas DataFrame 数据类型"
    data_raw = data_raw if torch.cuda.is_available() else data_raw.head(400)
    train_texts, val_texts, train_labels, val_labels, train_sensitive, val_sensitive = train_test_split(
        data_raw['text'], data_raw['label'], data_raw['sensitive'], test_size=0.2, stratify=data_raw['sensitive'])
    train_texts = train_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)
    train_sensitive = train_sensitive.reset_index(drop=True)
    val_sensitive = val_sensitive.reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if status == "train":
        model, tokenizer = load_model_Classification(model_name, data_raw['label'].max() + 1)
        model.to(device)

        train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=32)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        epochs = 3
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, device)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        report = test(model, val_loader, device, val_sensitive.to_numpy())
        print("model:", model_name, "dataset:", dataset)
        print(report)

        path = '/home/hongjixu/llm-bias/' + model_name + '_' + dataset
        model.save_pretrained(path)
    else:
        model, tokenizer = load_model_sequence_pretrain(path=model_name, name=name)
        model.to(device)
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=16)
        report = test(model, val_loader, device, val_sensitive.to_numpy())
        print(report)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to process model and dataset')

    # Add an argument for the model, this is a required argument
    parser.add_argument('--model', type=str, required=False, help='The name of the model to use')

    # Add an argument for the dataset, this is also a required argument
    parser.add_argument('--dataset', type=str, required=False, help='The name of the dataset to use')

    # Add an argument for the status, this is also a required argument
    parser.add_argument('--status', type=str, required=False, help='The status of llm')

    # Add an argument for the name of pre-trained model, this is also a required argument
    parser.add_argument('--name', type=str, required=False, help='The name of pre-trained model')

    # Parse the arguments
    args = parser.parse_args()

    main(args.model, args.dataset, args.status, args.name)
