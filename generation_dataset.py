import argparse
import json
import multiprocessing
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from dataset import load_realtoxic
from metrics import toxicity


def load_categories(file_path):
    """Load categories dictionary from a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)


def worker(worker_id, data_slice, categories_dict, sentence_threshold,
           prompt_length, results_queue):
    results = []
    for text in tqdm(data_slice, desc=f"Process {worker_id}"):
        results.extend(
            process_text(text, categories_dict, sentence_threshold,
                         prompt_length))
    results_queue.put(results)


def worker_prompt(worker_id, prompts, sentences, categories_dict, toxicitys,
                  results_queue):
    results = []
    for prompt, sentence, toxicity in tqdm(zip(prompts, sentences, toxicitys),
                                 total=len(prompts),
                                 desc=f"Process {worker_id}"):
        original_words = word_tokenize(prompt)
        prompt = ' '.join(original_words)
        results.extend(
            classify_prompt_based_on_category(prompt, sentence,
                                              categories_dict, toxicity))
    results_queue.put(results)


def process_text(text, categories_dict, sentence_threshold, prompt_length):
    """Process text to extract prompts based on categories."""
    results = []
    sentences = sent_tokenize(text)  # Split text into sentences
    for sentence in sentences:
        original_words = word_tokenize(sentence)  # Tokenize the sentence
        if len(original_words) >= sentence_threshold:
            prompt = ' '.join(
                original_words[:min(prompt_length, len(original_words))])
            results.extend(
                classify_prompt_based_on_category(prompt, sentence,
                                                  categories_dict))
    return results


def classify_prompt_based_on_category(prompt: str, sentence: str,
                                      categories_dict: dict) -> list:
    """
    Classify the given prompt into categories based on the provided categories dictionary.

    :param prompt: The prompt string to classify.
    :param categories_dict: A dictionary with domain as keys and another dictionary as values,
                            where the inner dictionary has category as keys and list of keywords as values.
    :return: A list of dictionaries, where each dictionary contains the domain, category, and the prompt.
    """
    results = []  # List to store the result
    # Convert prompt to lowercase and split into words
    words_lower = prompt.lower().split()
    # Generate possible phrases from the prompt words
    possible_phrases_lower = [
        ' '.join(words_lower[i:j + 1]) for i in range(len(words_lower))
        for j in range(i, min(i + 2, len(words_lower)))
    ]

    # Iterate through categories dictionary to find matching categories
    for domain, categories in categories_dict.items():
        flag = 0
        for category, keywords in categories.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check if the keyword matches any phrase from the prompt
                if any(keyword_lower == phrase
                       for phrase in possible_phrases_lower):
                    flag += 1
                    new_row = {
                        "domain": domain,
                        "category": category,
                        "texts": sentence,
                        "prompts": prompt,
                    }
                    break
            if flag > 1:
                break
        if flag == 1:
            results.append(new_row)
    return results


def load_dataset_chunks(dataset_name, num_processes):
    """Load the specified dataset and split it into chunks for multiprocessing."""
    if dataset_name == "jigsaw":
        dataset = load_dataset("SetFit/toxic_conversations")['train']
        data = dataset.to_pandas()
        # data = data[data['label'] == 1]
        data_chunks = np.array_split(data['text'], num_processes)
        file_name = "jigsaw_new.jsonl"

    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        data = pd.DataFrame()
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df['split'] = split
            data = pd.concat([data, df], ignore_index=True)
        data_chunks = np.array_split(data['text'], num_processes)
        file_name = "imdb_new.jsonl"

    elif dataset_name == "wiki_toxic":
        dataset = load_dataset("SetFit/toxic_conversations")
        data = pd.DataFrame()
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df['split'] = split
            data = pd.concat([data, df], ignore_index=True)
        data_chunks = np.array_split(data['text'], num_processes)
        file_name = "wikitoxic_new.jsonl"

    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0')['train']
        data = dataset.to_pandas()
        data_chunks = np.array_split(data['article'], num_processes)
        file_name = "cnn_dailymail_new.jsonl"

    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", 'wikitext-103-raw-v1')['train']
        data = dataset.to_pandas()
        data_chunks = np.array_split(data['text'], num_processes)
        file_name = "wikitext_new.jsonl"
        
    elif dataset_name == "bookcorpus":
        dataset = load_dataset('bookcorpus')['train']
        data = dataset.to_pandas()
        data_chunks = np.array_split(data['text'], num_processes)
        file_name = "bookcorpus_new.jsonl"

    elif dataset_name == "realtoxic":
        data = load_realtoxic()
        data_chunks = np.array_split(data, num_processes)
        file_name = "realtoxic_new.jsonl"

    else:
        raise ValueError("Unknown dataset")

    return data_chunks, file_name


def post_process(data: pd.DataFrame):
    max_samples = 3000
    random_state = 42
    all_data = pd.DataFrame()
    unique_domains = data['domain'].unique()

    for domain in unique_domains:
        domain_data = data[data['domain'] == domain]
        unique_categories = domain_data['category'].unique()
        domain_balanced = []

        for category in unique_categories:
            category_data = domain_data[domain_data['category'] == category]
            sample_size = min(len(category_data),
                              int(max_samples / len(unique_categories)))
            if len(category_data) > 0:
                sampled_data = category_data.sample(n=sample_size,
                                                    random_state=random_state)
                domain_balanced.append(sampled_data)

        all_data = pd.concat([all_data, pd.concat(domain_balanced)])
    return all_data


def calculate_scores(df, toxicity_evaluator, regard_evaluator):
    input_texts = df['prompts'].tolist()
    toxicity_results = toxicity_evaluator.compute(predictions=input_texts)
    toxicity_scores = toxicity_results["toxicity"]

    regard_results = regard_evaluator.compute(data=input_texts)
    regard_scores = [
        d[0]['score'] for d in regard_results['regard'] for l in d
        if l['label'] == 'negative'
    ]

    df['toxicity'] = toxicity_scores
    df['regard'] = regard_scores
    df['total_score'] = df['toxicity'] + df['regard']

    return df


def select_top_2k_per_category(df, file_path):
    toxicity_evaluator = evaluate.load("toxicity", module_type="measurement")
    regard_evaluator = evaluate.load("regard", module_type="measurement")

    df = calculate_scores(df, toxicity_evaluator, regard_evaluator)

    domain_category_counts = df.groupby(
        'domain')['category'].nunique().reset_index(name='unique_categories')
    domain_category_counts['per_category_quota'] = 1000 
    # 2000 // domain_category_counts['unique_categories']
    df = df.merge(domain_category_counts[['domain', 'per_category_quota']],
                  on='domain')

    print(f"Intermediate results are stored in {file_path}")
    df.to_json(file_path, orient='records', lines=True)

    def select_by_quota(group_df):
        quota = int(group_df['per_category_quota'].iloc[0])
        return group_df.nlargest(min(quota, len(group_df)), 'total_score')

    top_2k_per_domain = df.groupby(['domain', 'category'],
                                   group_keys=False).apply(select_by_quota)

    return top_2k_per_domain

def resample_large_domains(dataframe, threshold=60000, random_seed=42):
    """
    """
    domain_counts = dataframe['domain'].value_counts()
    large_domains = domain_counts[domain_counts > threshold].index
    filtered_df = dataframe[dataframe['domain'].isin(large_domains)]
    
    def resample_domain_group(group, target_size, seed):
        category_counts = group['category'].value_counts(normalize=True)
        resampled_group = group.groupby('category', group_keys=False).apply(
            lambda x: x.sample(n=int(target_size * category_counts.loc[x.name]), random_state=seed, replace=True)
        )
        return resampled_group
    
    resampled_df = filtered_df.groupby('domain', group_keys=False).apply(
        lambda x: resample_domain_group(x, threshold, random_seed)
    )
    resampled_df = resampled_df.reset_index(drop=True)
    
    cleaned_df = dataframe[~dataframe['domain'].isin(large_domains)]
    
    final_df = pd.concat([cleaned_df, resampled_df], ignore_index=True)
    
    return final_df

def main():
    parser = argparse.ArgumentParser(
        description='Process text to extract prompts based on categories.')
    parser.add_argument('--sentence_threshold',
                        type=int,
                        default=20,
                        help='Sentence length threshold')
    parser.add_argument('--prompt_length',
                        type=int,
                        default=10,
                        help='Prompt length in terms of number of words')
    parser.add_argument('--num_processes',
                        type=int,
                        default=60,
                        help='Number of processes to use')
    parser.add_argument('--dataset',
                        type=str,
                        default='bookcorpus',
                        help='Name of the dataset to use')
    parser.add_argument('--mode',
                        type=str,
                        default='text',
                        choices=['text', 'prompt'],
                        help='Mode of classification: text or prompt')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    wordlists_dir = script_dir / 'words'
    gender_dict = load_categories(wordlists_dir / "gender.yaml")
    occupation_dict = load_categories(wordlists_dir / "occupation.yaml")
    religion_dict = load_categories(wordlists_dir / "religion.yaml")
    age_dict = load_categories(wordlists_dir / "age.yaml")
    race_dict = load_categories(wordlists_dir / "race.yaml")
    bodyshaming_dict = load_categories(wordlists_dir / "bodyshaming.yaml")
    socioeconomic_dict = load_categories(wordlists_dir / "socioeconomic.yaml")
    lgbt_dict = load_categories(wordlists_dir / "lgbt.yaml")
    appearance_dict = load_categories(wordlists_dir / "appearance.yaml")
    class_dict = load_categories(wordlists_dir / "class.yaml")
    education_dict = load_categories(wordlists_dir / "education.yaml")
    disability_dict = load_categories(wordlists_dir / "disability.yaml")
    national_dict = load_categories(wordlists_dir / "national.yaml")

    categories_dict = {
        'gender': gender_dict,
        # 'occupation': occupation_dict,
        'religion': religion_dict,
        'age': age_dict,
        'race': race_dict,
        'bodyshaming': bodyshaming_dict,
        'socioeconomic': socioeconomic_dict,
        'lgbt': lgbt_dict,
        'appearance': appearance_dict,
        'class': class_dict,
        'education': education_dict,
        'disability': disability_dict,
        'national': national_dict
    }

    data_chunks, file_name = load_dataset_chunks(args.dataset,
                                                 args.num_processes)
    file_path = script_dir / 'data' / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.parent.exists():
        print(f"File {file_path}'s parent dir has been created successfully.")
    else:
        print(f"Failed to create the parent dir of file {file_path}.")

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    processes = []

    for i, data_chunk in enumerate(data_chunks):
        if args.mode == 'text':
            p = multiprocessing.Process(
                target=worker,
                args=(i, data_chunk, categories_dict, args.sentence_threshold,
                      args.prompt_length, results_queue))
        elif args.mode == 'prompt':
            p = multiprocessing.Process(target=worker_prompt,
                                        args=(i, data_chunk['prompts'],
                                              data_chunk['texts'],
                                              categories_dict, data_chunk['toxicity'], results_queue))
        processes.append(p)
        p.start()

    results = []
    for _ in range(args.num_processes):
        results.extend(results_queue.get())

    for p in processes:
        p.join()

    
    results_df = pd.DataFrame(results)
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())
    results_df = resample_large_domains(results_df)
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())
    
    if args.mode == "text":
        results_df = select_top_2k_per_category(results_df, file_path)
    
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())
    results_df.to_json(file_path, orient='records', lines=True)
    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()
