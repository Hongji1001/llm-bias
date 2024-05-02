from sklearn.model_selection import train_test_split
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from util import download_file
import torch
import pandas as pd 
from datasets import load_dataset
from folktables import ACSDataSource, ACSIncome

class CLMFineTuningDataset(Dataset):

    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Encode the texts with max_len
        text_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'labels': text_encoding['input_ids'].flatten(),
        }
  
        
class CLMGenerationDataset(Dataset):
    
    def __init__(self, prompts, texts, tokenizer, prompt_max_len=32, text_max_len=128):
        self.prompts = prompts
        self.texts = texts
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.text_max_len = text_max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        prompts = str(self.prompts[idx])

        # Encode the texts with max_len
        prompt_encoding = self.tokenizer.encode_plus(
            prompts,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        text_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.text_max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': prompt_encoding['input_ids'].flatten(),
            'attention_mask': prompt_encoding['attention_mask'].flatten(),
            'labels': text_encoding['input_ids'].flatten(),
        }
            

class MLMClassificationDataset(Dataset):

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
        

script_dir = Path(__file__).resolve().parent


def load_adult():
    data_raw = load_dataset("scikit-learn/adult-census-income")['train']
    data_raw = data_raw.to_pandas()

    data_raw['text'] = data_raw.iloc[:, :-1].apply(
        lambda row: ', '.join([f"{column}:{value}" for column, value in row.items()]), axis=1)
    data_raw['label'] = data_raw['income'].apply(lambda x: 0 if x == '<=50K' else 1)
    data_raw = data_raw[['text', 'label', 'sex']]
    data_raw['sex'] = data_raw['sex'].replace({'Female': 0, 'Male': 1}).astype(int)
    data_raw = data_raw.rename(columns={'sex': 'sensitive'})

    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=42)

    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'

    print(data_raw)

    return data_raw


def load_acsi():
    with open(script_dir / 'data' / 'config' / 'acs.yaml', 'r') as yaml_file:
        ACSIncome_categories = yaml.safe_load(yaml_file)
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    ca_data = ca_data.sample(frac=0.2, random_state=42)
    ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data, categories=ACSIncome_categories, dummies=False)
    ca_features = ca_features.rename(
        columns={
            'COW': 'Class of worker',
            'SCHL': 'Educational attainment',
            'MAR': 'Marital status',
            'OCCP': 'Occupation',
            'POBP': 'Place of birth',
            'RAC1P': 'Race'
        })
    ca_features['text'] = ca_features.apply(
        lambda row: ', '.join([f"{column}:{value}" for column, value in row.items()]), axis=1)
    ca_features['SEX'] = ca_features['SEX'].replace({'Female': 0, 'Male': 1}).astype(int)
    ca_features['label'] = ca_labels.replace({'False': 0, 'True': 1}).astype(int)
    ca_features = ca_features[['text', 'label', 'SEX']]
    ca_features = ca_features.rename(columns={'SEX': 'sensitive'})
    train_idx, test_idx = train_test_split(ca_features.index, test_size=0.2, random_state=42)

    ca_features['split'] = 'test'
    ca_features.loc[train_idx, 'split'] = 'train'

    print(ca_features)
    return ca_features


def load_bios():
    data_raw = load_dataset("LabHC/bias_in_bios")
    data = pd.DataFrame()
    for split in data_raw.keys():
        df = data_raw[split].to_pandas()
        df = df.sample(frac=0.1, random_state=42)
        df['split'] = split
        data = pd.concat([data, df], ignore_index=True)
    data_sampled = data.rename(columns={'hard_text': 'text', 'profession': 'label', 'gender': 'sensitive'})
    print(data_sampled)
    return data_sampled


def load_mdgender():
    data_raw = load_dataset("md_gender_bias", "funpedia")
    data = pd.DataFrame()
    for split in data_raw.keys():
        df = data_raw[split].to_pandas()
        df['split'] = split
        data = pd.concat([data, df], ignore_index=True)
    data_raw = data
    data_raw = data_raw[data_raw['gender'] != 0]
    data_raw['gender'] = data_raw['gender'].replace({1: 0, 2: 1}).astype(int)
    data_raw['label'] = data_raw['gender']
    data_raw = data_raw[['text', 'label', 'gender', 'split']]
    data_raw = data_raw.rename(columns={'gender': 'sensitive'})
    print(data_raw)
    return data_raw


def load_wikibias():
    file = download_file('https://docs.google.com/uc?export=download&id=1va3-3oBixdY4WEAOL3AvqcsGc5j2o34G',
                         cache_dir='~/.cache/wiki_bias',
                         filename='train.tsv')
    data_raw = pd.read_csv(file, delimiter='\t', header=None, names=['text', 'none', 'label'], index_col=False)
    with open(script_dir / 'words' / 'gender.yaml', 'r') as yaml_file:
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

    data_raw['sensitive'] = data_raw['text'].apply(
        lambda x: classify_text(x, binary_categories['male'], binary_categories['female']))
    data_raw = data_raw[data_raw['sensitive'] != 2]
    data_raw = data_raw[['text', 'label', 'sensitive']]
    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=42)

    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'
    print(data_raw)
    return data_raw


def load_wikitalk():
    data_raw = load_dataset("dirtycomputer/Wikipedia_Talk_Labels")['train']
    data_raw = data_raw.to_pandas()
    data_raw = data_raw[data_raw['comment'].apply(len) <= 1024]
    # data_raw = data_raw.sample(frac=0.2, random_state=42)
    with open(script_dir / 'words' / 'gender.yaml', 'r') as yaml_file:
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

    data_raw['sensitive'] = data_raw['comment'].apply(
        lambda x: classify_text(x, binary_categories['male'], binary_categories['female']))
    data_raw = data_raw[data_raw['sensitive'] != 2]
    data_raw = data_raw.rename(columns={'comment': 'text', 'attack': 'label'})
    data_raw['label'] = data_raw['label'].replace({False: 0, True: 1}).astype(int)
    data_raw = data_raw[['text', 'label', 'sensitive']]
    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=42)
    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'
    print(data_raw)
    return data_raw


def load_jigsaw():
    data_raw = pd.read_csv("data/identity_individual_annotations.csv")
    # data_sampled = data_raw.rename(columns={'hard_text': 'text', 'profession': 'label', 'gender': 'sensitive'})
    data_raw = data_raw[data_raw['gender'].isin(['male', 'female'])]
    data_raw.reset_index(drop=True, inplace=True)
    print(data_raw)
    return data_raw


def load_classification_dataset(dataset):
    if dataset == "adult":
        return load_adult()
    elif dataset == "acs":
        return load_acsi()
    elif dataset == "bios":
        return load_bios()
    elif dataset == "mdgender":
        return load_mdgender()
    elif dataset == "wikibias":
        return load_wikibias()
    elif dataset == "wikitalk":
        return load_wikitalk()
    else:
        raise NameError(dataset)