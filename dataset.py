import json
from typing import Literal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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


def load_adult(protected_group):
    if protected_group not in ['gender', 'race', 'age', 'national']:
        print(f"adult do not support protected group {protected_group}")
        return None
    
    data_raw = load_dataset("scikit-learn/adult-census-income")['train']
    data_raw = data_raw.to_pandas()

    data_raw['text'] = data_raw.iloc[:, :-1].apply(
        lambda row: ', '.join([f"{column}:{value}" for column, value in row.items()]), axis=1)
    data_raw['label'] = data_raw['income'].apply(lambda x: 0 if x == '<=50K' else 1)
    
    # Categorize age into four groups
    def categorize_age(age):
        if age <= 17:
            return 'minor'
        elif 18 <= age <= 65:
            return 'young'
        elif 66 <= age <= 79:
            return 'middle_aged'
        else:
            return 'elderly'
    
    def categorize_national(national):
        if national != 'United-States':
            return 'Non-United-States'
        return national
        
    if protected_group == 'gender':
        data_raw = data_raw.rename(columns={'sex': 'sensitive'})
    elif protected_group == 'race':
        data_raw = data_raw.rename(columns={'race': 'sensitive'})
    elif protected_group == 'age':
        data_raw['age_group'] = data_raw['age'].apply(categorize_age)
        data_raw = data_raw.rename(columns={'age_group': 'sensitive'})
    elif protected_group == 'national':
        data_raw['native.country'] = data_raw['native.country'].apply(categorize_national)
        data_raw = data_raw.rename(columns={'native.country': 'sensitive'})
        
    data_raw = data_raw[['text', 'label', 'sensitive']]
        
    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=0)
    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'
    print("loading adult")
    print(data_raw)
    return data_raw


def load_acsi(protected_group):
    if protected_group not in ['gender', 'race']:
        print(f"acsi do not support protected group {protected_group}")
        return None
    with open(script_dir / 'data' / 'config' / 'acs.yaml', 'r') as yaml_file:
        ACSIncome_categories = yaml.safe_load(yaml_file)
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    ca_data = ca_data.sample(frac=0.2, random_state=0)
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
    # ca_features['SEX'] = ca_features['SEX'].replace({'Female': 0, 'Male': 1}).astype(int)
    ca_features['label'] = ca_labels.replace({'False': 0, 'True': 1}).astype(int)
    if protected_group == 'gender':
        ca_features = ca_features.rename(columns={'SEX': 'sensitive'})
    elif protected_group == 'race':
        ca_features = ca_features.rename(columns={'Race': 'sensitive'})
    ca_features = ca_features[['text', 'label', 'sensitive']]
    train_idx, test_idx = train_test_split(ca_features.index, test_size=0.2, random_state=0)

    ca_features['split'] = 'test'
    ca_features.loc[train_idx, 'split'] = 'train'
    print("loading acsi")
    print(ca_features)
    return ca_features


def load_bios():
    data_raw = load_dataset("LabHC/bias_in_bios")
    data = pd.DataFrame()
    for split in data_raw.keys():
        df = data_raw[split].to_pandas()
        df = df.sample(frac=0.1, random_state=0)
        df['split'] = split
        data = pd.concat([data, df], ignore_index=True)
    data_sampled = data.rename(columns={'hard_text': 'text', 'profession': 'label', 'gender': 'sensitive'})
    print(data_sampled)
    return data_sampled


def load_mdgender(protected_group):
    if protected_group not in ['gender']:
        print(f"mdgender do not support protected group {protected_group}")
        return None
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


def load_wikibias(protected_group):
    if protected_group not in ['gender']:
        print(f"wikibias do not support protected group {protected_group}")
        return None
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
    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=0)

    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'
    print("loading wikibias")
    print(data_raw)
    return data_raw


def load_wikitalk(protected_group):
    if protected_group not in ['gender']:
        print(f"wikitalk do not support protected group {protected_group}")
        return None
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
    train_idx, test_idx = train_test_split(data_raw.index, test_size=0.2, random_state=0)
    data_raw['split'] = 'test'
    data_raw.loc[train_idx, 'split'] = 'train'
    print("loading wikitalk")
    print(data_raw)
    return data_raw

def load_jigsaw(protected_group):
    if protected_group not in ['gender', 'race', 'religion', 'sexual_orientation']:
        print(f"jigsaw do not support protected group {protected_group}")
        return None
    file_path = Path('data/jigsaw.jsonl')
    if file_path.is_file():
        data_raw = pd.read_json(file_path, lines=True)
        # TODO: check whether split different domain
        data_raw = data_raw[data_raw['domain'] == protected_group]
        print(data_raw)
        return data_raw
    else:
        print("Generating jigsaw.jsonl for the first time")
        
    all_data = pd.read_csv('data/all_data.csv')
    
    # Initialize a list to collect the rows for each domain
    processed_data = []

    # Define domains and categories
    domains = {
        "gender": ['male', 'female'],
        # "disability": ['psychiatric_or_mental_illness', 'intellectual_or_learning_disability', 'physical_disability'],
        "race": ['black', 'white', 'asian'],
        "religion": ['christian', 'muslim', 'jewish'],
        "sexual_orientation" : ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual'],
    }

    # Process the data for each domain
    for index, row in tqdm(all_data.iterrows(), total=all_data.shape[0], desc="Processing rows"):
        for domain, categories in domains.items():
            # Filter the columns based on the domain categories
            domain_values = {category: row[category] for category in categories}
            valid_categories = [key for key, value in domain_values.items() if value >= 0.5]

            if len(valid_categories) == 1:
                # Create a new row with the desired format
                new_row = {
                    "domain": domain,
                    "sensitive": valid_categories[0],
                    "split": row['split'],
                    "text": row['comment_text'],
                    "label": 1 if row['toxicity'] >= 0.5 else 0
                }
                processed_data.append(new_row)

    # Convert the list to a DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Save the DataFrame to a JSON Lines file
    processed_df.to_json('data/jigsaw.jsonl', orient='records', lines=True)
    
    # Calculate and print aggregated statistics
    aggregation = processed_df.groupby(['split', 'domain', 'sensitive', 'label']).size().reset_index(name='count')
    train_counts = aggregation[aggregation['split'] == 'train']
    test_counts = aggregation[aggregation['split'] == 'test']
    print("loading jigsaw")
    print("Train split counts:")
    print(train_counts)
    print("\nTest split counts:")
    print(test_counts)
    processed_df = processed_df[processed_df['domain'] == protected_group]
    print(processed_df)
    return processed_df

def load_equibench(protected_group):
    print("loading equibench", protected_group)
    all_data = pd.read_json('data/equibench.jsonl', lines=True)
    domain_data = all_data[all_data['domain'] == protected_group]
    if len(domain_data) == 0:
        print("equibench do not support type", protected_group)
        return None
    train_idx, test_idx = train_test_split(domain_data.index, test_size=0.2, random_state=0)
    domain_data['split'] = 'test'
    domain_data.loc[train_idx, 'split'] = 'train'
    
    print(domain_data)
    return domain_data


def load_classification_dataset(dataset, protected_group):
    if dataset == "adult":
        return load_adult(protected_group)
    elif dataset == "acs":
        return load_acsi(protected_group)
    elif dataset == "bios":
        return load_bios()
    elif dataset == "mdgender":
        return load_mdgender(protected_group)
    elif dataset == "wikibias":
        return load_wikibias(protected_group)
    elif dataset == "wikitalk":
        return load_wikitalk(protected_group)
    elif dataset == "jigsaw":
        return load_jigsaw(protected_group)
    elif dataset == "equibench":
        return load_equibench(protected_group)
    else:
        raise NameError(dataset)
    

def load_realtoxic():
    data_raw = load_dataset("allenai/real-toxicity-prompts")
    data = pd.DataFrame()
    for split in data_raw.keys():
        df = data_raw[split].to_pandas()
        df['split'] = split    # 添加一个列来指明当前行属于哪个数据集分区
        data = pd.concat([data, df], ignore_index=True)    # 合并DataFrame
    data_raw = data
    # data_raw = data_raw[data_raw['challenging'] == False]
    data_raw = data_raw[['prompt', 'continuation', 'challenging']]
    data_raw['toxicity'] = data_raw['prompt'].apply(lambda x: x['toxicity'])
    data_raw['prompt'] = data_raw['prompt'].apply(lambda x: x['text'])
    data_raw['continuation'] = data_raw['continuation'].apply(lambda x: x['text'])
    data_raw['continuation'] = data_raw['prompt'] + data_raw['continuation']
    def extract_toxicity(json_str):
        json_obj = json.loads(json_str)
        return json_obj.get('toxicity', None)
    data_raw = data_raw.rename(columns={'continuation': 'texts', 'prompt': 'prompts', 'challenging': 'sensitive'})
    print(data_raw)
    return data_raw