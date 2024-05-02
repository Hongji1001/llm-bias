from pathlib import Path
import pandas as pd
import os
data_list = ["cnn_dailymail_2k", "bold", "imdb_2k", "realtoxic_2k", "jigsaw_2k", "wikitext_2k", "wikitoxic_2k"]
script_dir = Path(__file__).resolve().parent
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix
from dataset import MLMClassificationDataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import load_adult, load_bios, load_mdgender, load_wikitalk, load_acsi, load_wikibias, load_jigsaw

def gen_selfbias():
    for name in data_list:  
        output_path = f'gpt2_{name}_selfbias.jsonl'
        name += '.jsonl'
        filename = script_dir/ 'data' / name
        print(filename)
        print(os.path.exists(filename))
        data_raw = pd.read_json(filename, lines=True)
        data_new = pd.read_json(output_path, lines=True)
        data_raw = data_raw.rename(columns={'sensitive': 'domain'})
        data_new['sensitives'] = data_raw['domain']
        data_new['category'] = data_raw['category']
        prompt = data_new.get('prompt', '')
        data = data_new['continuations'].apply(lambda x: x[0]['text'] if x else '')
        print(data)
        data_new['completions'] = prompt + data
        data_new['completions_split'] = data_new['completions'].apply(lambda x: x.split())
        print(data_new)
        data_new.to_json(f'outputs/completions/{output_path}', orient='records', lines=True)


def json2jsonl():
    for name in data_list:
        name += '.json'
        filename = script_dir/ 'data' / name
        data_raw = pd.read_json(filename)
        print(data_raw)
        data_raw.to_json(f'data/{name}'+'l', orient='records', lines=True)


def predict(texts, labels, model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.cuda()
    prompt_dataset = MLMClassificationDataset(texts, labels, tokenizer)
    prompt_loader = DataLoader(prompt_dataset, 16)
    predictions = []
    with torch.no_grad():
        for batch in prompt_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            predictions.extend(prediction)
    return predictions


def expr():
    datalist = ['age', 'appearance', 'bodyshaming', 'lgbt', 'racial', 'socioeconomic']
    model_names = ["D1V1DE/bias-detection", "valurank/distilroberta-bias"]
    for bias_type in datalist:
        for model_name in model_names:  
            bias = pd.read_json(bias_type+'_bias.jsonl', lines=True)
            non_bias = pd.read_json(bias_type+'_non_bias.jsonl', lines=True)
            data = pd.concat([bias, non_bias], ignore_index=True)
            data['label'] = data['classification'].map({'bias': 1, 'non_bias': 0})
            print(len(data))
            predictions = predict(data['output'], data['label'], model_name)
            true_labels = data['label']
            conf_matrix = confusion_matrix(true_labels, predictions)
            print(f"{model_name} {bias_type} confusion matric:")
            print(conf_matrix)


def combine_data():
    type_ = 'socioeconomic'
    data_bias = pd.read_json(f'{type_}_bias.jsonl', lines=True)
    data_nonbias = pd.read_json(f'{type_}_non_bias.jsonl', lines=True)
    data_combined = pd.concat([data_bias, data_nonbias])
    file_path = Path(__file__).resolve().parent / 'data' / f'gen_{type_}.jsonl'
    data_combined = data_combined.rename(columns={'output': 'text', 'classification': 'label'})
    data_combined['label'] = data_combined['label'].replace({'bias': 1, 'non_bias': 0})
    data_combined['sensitive'] = data_combined['label']
    train_idx, test_idx = train_test_split(data_combined.index, test_size=0.2, random_state=42)
    data_combined['split'] = 'test'
    data_combined.loc[train_idx, 'split'] = 'train'
    data_combined.to_json(file_path, orient='records', lines=True)
    print(data_combined)
    pass


def MWU():
    import numpy as np
    from scipy.stats import mannwhitneyu

    # Sample data
    data1 = np.random.uniform(size=1000)
    data2 = 1 - data1
    print(data1[:10], data2[:10])
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    PosAvgEG = 0.5 - u_statistic/ (len(data1) * len(data2))
    print("Mann-Whitney U statistic:", u_statistic)
    print("P-value:", p_value)
    print("PosAvgEG:", PosAvgEG)


# data_raw = data_raw[data_raw['sensitive'] == 'gender']
# data_raw['prompt'] = data_raw['prompts'].apply(lambda x: {'text': x})
# # Add a 'challenge' column with a placeholder value
# data_raw['challenging'] = True
# jsonl_str = data_raw.to_json(orient='records', lines=True)
# # The resulting JSON Lines formatted string
# file_path += 'l'
# file_path = script_dir.parent / 'data' / file_path
# with open(file_path, 'w') as f:
#     f.write(jsonl_str)

if __name__ == '__main__':
    load_jigsaw()