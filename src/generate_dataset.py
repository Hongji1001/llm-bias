import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from dataset import load_realtoxic


def load_categories(file_path):
    """Load categories dictionary from a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)


def worker(worker_id, data_slice, categories_dict, sentence_threshold, prompt_length, results_queue):
    results = []
    for text in tqdm(data_slice, desc=f"Process {worker_id}"):
        results.extend(process_text(text, categories_dict, sentence_threshold, prompt_length))
    results_queue.put(results)


def worker_prompt(worker_id, prompts, sentences, categories_dict, results_queue):
    results = []
    for prompt, sentence in tqdm(zip(prompts, sentences), total=len(prompts), desc=f"Process {worker_id}"):
        original_words = word_tokenize(prompt)
        prompt = ' '.join(original_words)
        results.extend(classify_prompt_based_on_category(prompt, sentence, categories_dict))
    results_queue.put(results)


def process_text(text, categories_dict, sentence_threshold, prompt_length):
    """Process text to extract prompts based on categories."""
    results = []
    sentences = sent_tokenize(text)    # Split text into sentences
    for sentence in sentences:
        original_words = word_tokenize(sentence)    # Tokenize the sentence
        if len(original_words) >= sentence_threshold:
            prompt = ' '.join(original_words[:min(prompt_length, len(original_words))])
            results.extend(classify_prompt_based_on_category(prompt, sentence, categories_dict))
    return results


def classify_prompt_based_on_category(prompt: str, sentence: str, categories_dict: dict) -> list:
    """
    Classify the given prompt into categories based on the provided categories dictionary.

    :param prompt: The prompt string to classify.
    :param categories_dict: A dictionary with domain as keys and another dictionary as values,
                            where the inner dictionary has category as keys and list of keywords as values.
    :return: A list of dictionaries, where each dictionary contains the domain, category, and the prompt.
    """
    results = []    # List to store the result
    # Convert prompt to lowercase and split into words
    words_lower = prompt.lower().split()
    # Generate possible phrases from the prompt words
    possible_phrases_lower = [
        ' '.join(words_lower[i:j + 1]) for i in range(len(words_lower)) for j in range(i, min(i + 2, len(words_lower)))
    ]

    # Iterate through categories dictionary to find matching categories
    for domain, categories in categories_dict.items():
        flag = 0
        for category, keywords in categories.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check if the keyword matches any phrase from the prompt
                if any(keyword_lower == phrase for phrase in possible_phrases_lower):
                    flag += 1
                    new_row = {"domain": domain, "category": category, "texts": sentence, "prompts": prompt}
                    break
            if flag > 1:
                break
        if flag == 1:
            results.append(new_row)
    return results


def post_process(df: pd.DataFrame):
    # Step 1: Filter for 'gender' domain data
    gender_data = df[df['domain'] == 'gender']

    # Step 2: Balance 'male' and 'female' counts
    male_data = gender_data[gender_data['category'] == 'male']
    female_data = gender_data[gender_data['category'] == 'female']
    min_count = min(len(male_data), len(female_data))
    balanced_male = male_data.sample(n=min_count)
    balanced_female = female_data.sample(n=min_count)
    balanced_gender = pd.concat([balanced_male, balanced_female])

    # Step 3: Randomly select 20% from balanced data
    sampled_gender = balanced_gender.sample(frac=0.1)

    # Step 4: Select non-'gender' domain data
    non_gender_data = df[df['domain'] != 'gender']

    # Step 5: Combine the datasets
    final_df = pd.concat([sampled_gender, non_gender_data])
    return final_df


def main():
    parser = argparse.ArgumentParser(description='Process text to extract prompts based on categories.')
    parser.add_argument('--file_name', type=str, help='Path to the YAML file containing categories dictionary')
    parser.add_argument('--sentence_threshold', type=int, default=20, help='Sentence length threshold')
    parser.add_argument('--prompt_length', type=int, default=10, help='Prompt length in terms of number of words')
    parser.add_argument('--num_processes', type=int, default=20, help='Number of processes to use')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    wordlists_dir = script_dir.parent / 'words'
    gender_dict = load_categories(wordlists_dir / "gender.yaml")
    occupation_dict = load_categories(wordlists_dir / "profession.yaml")
    religion_dict = load_categories(wordlists_dir / "religion.yaml")
    age_dict = load_categories(wordlists_dir / "age.yaml")
    race_dict = load_categories(wordlists_dir / "race.yaml")

    categories_dict = {
        # 'gender': gender_dict,
    # 'occupation': occupation_dict,
    # 'religion': religion_dict,
    # 'age': age_dict
    'race': race_dict
    }
    # dataset = load_dataset("SetFit/toxic_conversations")['train']
    # data = dataset.to_pandas()
    # data = data[data['label'] == 1]
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "jigsaw_toxic.json"

    # dataset = load_dataset("imdb")['train']
    # data = dataset.to_pandas()
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "imdb.json"

    # dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic")['train']
    # data = dataset.to_pandas()
    # data_chunks = np.array_split(data['comment_text'], args.num_processes)
    # file_name = "wikitoxic.json"

    # dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0')['train']
    # data = dataset.to_pandas()
    # data = data.sample(frac=0.1, random_state=42)
    # data_chunks = np.array_split(data['article'], args.num_processes)
    # file_name = "cnn_dailymail.json"

    # dataset = load_dataset("wikitext", 'wikitext-103-raw-v1')['train']
    # data = dataset.to_pandas()
    # data = data.sample(frac=0.1, random_state=42)
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "wikitext.json"

    data_bias = pd.read_json(
        Path(__file__).resolve().parent.parent / 'racial_bias.jsonl', lines=True)
    data_nonbias = pd.read_json(
        Path(__file__).resolve().parent.parent / 'racial_non_bias.jsonl', lines=True)
    data_combined = pd.concat([data_bias, data_nonbias])
    print(len(data_combined))
    print(data_combined)
    # data = load_realtoxic()
    # print(data)
    data_chunks = np.array_split(data_combined, args.num_processes)
    file_name = Path(
        __file__).resolve().parent.parent / 'data' / "gen_race.jsonl"

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    processes = []
    for i, data_chunk in enumerate(data_chunks):
        # for categorize from corpus/texts
        # p = multiprocessing.Process(target=worker,
        #                             args=(i, data_chunk, categories_dict, args.sentence_threshold, args.prompt_length,
        #                                   results_queue))
        # for categorize from prompts
        p = multiprocessing.Process(target=worker_prompt,
                                    args=(i, data_chunk['output'],
                                          data_chunk['output'], categories_dict,
                                          results_queue))
        processes.append(p)
        p.start()

    # Collect results from each process
    results = []
    for _ in range(args.num_processes):
        results.extend(results_queue.get())

    # Wait for all processes to complete
    for p in processes:
        p.join()

    results_df = pd.DataFrame(results)
    results_df = post_process(results_df)
    print(results_df)
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())

    # Save results directly
    file_path = script_dir.parent / 'data' / file_name
    results_df.to_json(file_path, orient='records', lines=True, indent=4)
    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()
