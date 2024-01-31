import torch
import argparse
from tqdm import tqdm
from dataset import AdultDataset, data_loader
from model import load_model_Classification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train(model, data_loader, optimizer, device):
    model.train()
    total_loss, total_accuracy = 0, 0

    for batch in tqdm(data_loader, unit='batch'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(data_loader), accuracy


def main(model_name, dataset):
    data_raw = data_loader(dataset)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data_raw['text'], data_raw['label'], test_size=0.2)
    train_texts = train_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)

    model, tokenizer = load_model_Classification(
        model_name, data_raw['label'].nunique())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = AdultDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = AdultDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(
            f'Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

    path = '/home/hx84/llm-bias/checkpoint/' + model_name + '_' + dataset
    model.save_pretrained(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')

    # Add an argument for the model, this is a required argument
    parser.add_argument('--model', type=str, required=True,
                        help='The name of the model to use')

    # Add an argument for the dataset, this is also a required argument
    parser.add_argument('--dataset', type=str, required=True,
                        help='The name of the dataset to use')

    # Parse the arguments
    args = parser.parse_args()

    main(args.model, args.dataset)
