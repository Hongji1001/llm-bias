import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from folktables import ACSDataSource, ACSIncome

script_dir = Path(__file__).resolve().parent

class AdultDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_adult():
    data_raw = load_dataset("scikit-learn/adult-census-income")['train']
    data_raw = data_raw.to_pandas()
    data_raw['text'] = data_raw.iloc[:, :-1].apply(lambda row: ', '.join(
        [f"{column}:{value}" for column, value in row.items()]), axis=1)
    data_raw['label'] = data_raw['income'].apply(
        lambda x: 0 if x == '<=50K' else 1)
    data_raw = data_raw[['text', 'label', 'sex']]
    data_raw['sex'] = data_raw['sex'].replace({'Female': 0, 'Male': 1}).astype(int)
    data_raw = data_raw.rename(
        columns={'sex': 'sensitive'})
    return data_raw


def load_acs_i():
    with open(script_dir / 'acs.yaml', 'r') as yaml_file:
        ACSIncome_categories = yaml.safe_load(yaml_file)
    data_source = ACSDataSource(
        survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)

    ca_features, ca_labels, _ = ACSIncome.df_to_pandas(
        ca_data, categories=ACSIncome_categories, dummies=False)
    ca_features = ca_features.rename(columns={'COW': 'Class of worker', 'SCHL': 'Educational attainment',
                                              'MAR': 'Marital status', 'OCCP': 'Occupation', 'POBP': 'Place of birth', 'RAC1P': 'Race'})
    ca_features['text'] = ca_features.apply(lambda row: ', '.join(
        [f"{column}:{value}" for column, value in row.items()]), axis=1)
    ca_features['SEX'] = ca_features['SEX'].replace({'Female': 0, 'Male': 1}).astype(int)
    ca_features['label'] = ca_labels
    ca_features = ca_features[['text', 'label', 'SEX']]
    ca_features = ca_features.rename(
        columns={'SEX': 'sensitive'})
    return ca_features


def load_bios():
    data_raw = load_dataset("LabHC/bias_in_bios")['dev']
    data_raw = data_raw.to_pandas()
    # data_raw = data_raw[['hard_text', 'profession']]
    data_raw = data_raw.rename(
        columns={'hard_text': 'text', 'profession': 'label', 'gender': 'sensitive'})
    print(data_raw)
    return data_raw


def load_md_gender():
    data_raw = load_dataset("md_gender_bias", "funpedia")['train']
    data_raw = data_raw.to_pandas()
    data_raw = data_raw[data_raw['gender'] != 0]
    data_raw['gender'] = data_raw['gender'].replace({1: 0, 2: 1}).astype(int)
    data_raw['label'] = data_raw['gender']
    data_raw = data_raw[['text', 'label', 'gender']]
    data_raw = data_raw.rename(
        columns={'gender': 'sensitive'})
    print(data_raw)
    return data_raw


def load_wikibias():
    data_raw = pd.read_csv('/home/hx84/llm-bias/dataset/wikibias/class_binary/train.tsv', delimiter='\t',
                           header=None, names=['text', 'none', 'label'], index_col=False)
    with open(script_dir / 'gender.yaml', 'r') as yaml_file:
        binary_categories = yaml.safe_load(yaml_file)
        
    def classify_text(text, category_0_words, category_1_words):
        male_present = any(word in text.split() for word in category_0_words)
        female_present = any(word in text.split() for word in category_1_words)

        if male_present and not female_present:
            return 1
        elif female_present and not male_present:
            return 0
        else:
            return 2
    data_raw['sensitive'] = data_raw['text'].apply(lambda x: classify_text(x, binary_categories['male'], binary_categories['female']))
    data_raw = data_raw[data_raw['sensitive'] != 2]
    data_raw = data_raw[['text', 'label', 'sensitive']]
    print(data_raw)
    return data_raw


def load_wiki_talk():
    data_raw = load_dataset("dirtycomputer/Wikipedia_Talk_Labels")['train']
    data_raw = data_raw.to_pandas()
    data_raw = data_raw[data_raw['comment'].apply(len) <= 1024]
    # data_raw = data_raw.sample(frac=0.2, random_state=42)
    with open(script_dir / 'gender.yaml', 'r') as yaml_file:
        binary_categories = yaml.safe_load(yaml_file)
        
    def classify_text(text, category_0_words, category_1_words):
        male_present = any(word in text.split() for word in category_0_words)
        female_present = any(word in text.split() for word in category_1_words)

        if male_present and not female_present:
            return 1
        elif female_present and not male_present:
            return 0
        else:
            return 2
    data_raw['sensitive'] = data_raw['comment'].apply(lambda x: classify_text(x, binary_categories['male'], binary_categories['female']))
    data_raw = data_raw[data_raw['sensitive'] != 2]
    data_raw = data_raw.rename(
        columns={'comment': 'text', 'attack': 'label'})
    data_raw['label'] = data_raw['label'].replace({False: 0, True: 1}).astype(int)
    data_raw = data_raw[['text', 'label', 'sensitive']]
    print(data_raw)
    return data_raw


def load_crows_pair(metric_type):
    data_raw = load_dataset("crows_pairs")['test']
    if metric_type == "CPS":
        data_raw = data_raw.to_pandas()
        data_raw = data_raw.rename(columns={'stereo_antistereo': 'direction'})[['sent_more', 'sent_less',
                                                                               'direction', 'bias_type']]
    return data_raw


def load_stereo_set(metric_type):
    data_raw = load_dataset("stereoset", "intrasentence")['validation']
    if metric_type == "CPS":
        sent_more = []
        sent_less = []
        bias_type = []
        for item in data_raw:
            sentences = item['sentences']
            sent_more.append(sentences['sentence'][0])
            sent_less.append(sentences['sentence'][2])
            bias_type.append(item['bias_type'])
        data_raw = pd.DataFrame({
            'sent_more': sent_more,
            'sent_less': sent_less,
            'direction': [0] * len(sent_more),
            'bias_type': bias_type
        })
    return data_raw


def load_wino_bias(metric_type):
    data_1_pro = load_dataset("wino_bias", "type1_pro")
    data_1_anti = load_dataset("wino_bias", "type1_anti")
    if metric_type == "CPS":
        sent_more = []
        sent_less = []
        for item in data_1_pro['validation']:
            sent_more_pro = " ".join(item['tokens'])
            sent_more.append(sent_more_pro)
        for item in data_1_pro['test']:
            sent_more_pro = " ".join(item['tokens'])
            sent_more.append(sent_more_pro)
        for item in data_1_anti['validation']:
            sent_less_anti = " ".join(item['tokens'])
            sent_less.append(sent_less_anti)
        for item in data_1_anti['test']:
            sent_less_anti = " ".join(item['tokens'])
            sent_less.append(sent_less_anti)
        data_raw = pd.DataFrame({
            'sent_more': sent_more,
            'sent_less': sent_less,
            'direction': [0] * len(sent_more),
            'bias_type': '' * len(sent_more)
        })
    return data_raw


def load_bias_nli(metric_type):
    data_raw = pd.read_csv(
        "../dataset/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/gender_bias_cleaned_output.csv")
    if metric_type == "CPS":
        data_raw = pd.DataFrame({
            'sent_more': data_raw.iloc[:, -2],
            'sent_less': data_raw.iloc[:, -1],
            'direction': [0] * len(data_raw),
            'bias_type': ['gender'] * len(data_raw)
        })
    return data_raw


def load_win_queer(metric_type):
    data_raw = pd.read_csv(
        "../dataset/winoqueer/data/winoqueer_final.csv")
    if metric_type == "CPS":
        data_raw = pd.DataFrame({
            'sent_more': data_raw.iloc[:, -2],
            'sent_less': data_raw.iloc[:, -1],
            'direction': [0] * len(data_raw),
            'bias_type': data_raw.iloc[:, -4],
        })
    return data_raw


def data_loader(dataset="crows_pairs", metric="CPS"):
    if dataset == "crows_pairs":
        return load_crows_pair(metric)
    elif dataset == "stereo_set":
        return load_stereo_set(metric)
    elif dataset == "wino_bias":
        return load_wino_bias(metric)
    elif dataset == "bias_nli":
        return load_bias_nli(metric)
    elif dataset == "win_queer":
        return load_win_queer(metric)
    elif dataset == "adult":
        return load_adult()
    elif dataset == "acs":
        return load_acs_i()
    elif dataset == "bios":
        return load_bios()
    elif dataset == "mdgender":
        return load_md_gender()
    elif dataset == "wikibias":
        return load_wikibias()
    elif dataset == "wiki_talk":
        return load_wiki_talk()
    else:
        print(f"{dataset} is not supported now")


if __name__ == '__main__':
   load_md_gender()