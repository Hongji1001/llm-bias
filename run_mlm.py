import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import MLMClassificationDataset, load_classification_dataset
from metrics import PosAvgEG, eod, kld, spd, eod_v1, kld_v1, spd_v1
from torch.utils.data import DataLoader


def test_mlm_for_classification(model, tokenizer, dataset, outputpath):
    model.eval()
    device = 'cuda'
    model.to(device)
    batch_size = 16
    
    texts = dataset['text'].reset_index(drop=True)
    labels = dataset['label'].reset_index(drop=True)
    sensitive_list = dataset['sensitive'].reset_index(drop=True)
    
    test_dataset = MLMClassificationDataset(texts, labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size)
    
    predictions, true_labels, target_hat_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, unit='batch'):
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
    print(report)
    eod_result = eod(target_hat_list, np.array(true_labels), np.array(sensitive_list), 0.5)
    kld_result = kld(target_hat_list, np.array(sensitive_list))
    print("kld: ", kld_result)
    spd_result = spd(target_hat_list[:, 1], np.array(sensitive_list))
    print("spd: ", spd_result)
    # eod_result = eod_v1(target_hat_list, np.array(true_labels), np.array(sensitive_list), 0.5)
    # kld_result = kld_v1(target_hat_list, np.array(sensitive_list))
    # print("kld_1: ", kld_result)
    # spd_result = spd_v1(target_hat_list[:, 1], np.array(sensitive_list))
    # print("spd_v1: ", spd_result)
    PosAvgEG_result = PosAvgEG(target_hat_list, np.array(true_labels), np.array(sensitive_list))
    
    with open(outputpath, 'a') as f:
        f.write(report)
        for k,v in eod_result.items():
            f.write(f"EOD for category {k}: {v}%\n")
        f.write(f"kld: {kld_result}\n")
        f.write(f"spd: {spd_result}\n")
        for result in PosAvgEG_result:
            f.write(f"Identity: {result['identity']}\n")
            f.write(f"BPSN_AUC: {result['BPSN_AUC']}\n")
            f.write(f"PosAvgEG: {result['PosAvgEG']}\n")
            f.write(f"NegAvgEG: {result['NegAvgEG']}\n")


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
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.dataset_file is not None:
        train_dataset = pd.read_json(args.dataset_file, lines=True)
        num_labels = train_dataset['label'].nunique()
    elif args.dataset_name is not None:
        train_dataset = load_classification_dataset(args.dataset_name)
        train_dataset = train_dataset[train_dataset['split'] == 'test'].reset_index(drop=True)
        num_labels = train_dataset['label'].nunique()
    else:
        raise argparse.ArgumentError('give at least one in --dataset_file and --dataset_name')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    output_file_path = Path(__file__).resolve().parent / 'outputs'
    output_file_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_path / 'metrics'
    output_file_path.mkdir(parents=True, exist_ok=True)
    filename = output_file_path / f"eval_output_{args.dataset_name}_{args.tokenizer_name}.log"
    print(filename)
    test_mlm_for_classification(model, tokenizer, train_dataset, filename)