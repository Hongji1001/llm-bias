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


def process_text(text, categories_dict, sentence_threshold, prompt_length):
    """Process text to extract prompts based on categories."""
    results = []
    sentences = sent_tokenize(text)  # Split text into sentences
    for sentence in sentences:
        original_words = word_tokenize(sentence)  # Tokenize the sentence
        if len(original_words) >= sentence_threshold:
            prompt = ' '.join(
                original_words[:min(prompt_length, len(original_words))])
            words_lower = [
                word.lower() for word in
                original_words[:min(prompt_length, len(original_words))]
            ]
            possible_phrases_lower = [
                ' '.join(words_lower[i:j + 1]) for i in range(len(words_lower))
                for j in range(i, min(i + 2, len(words_lower)))
            ]

            for domain, categories in categories_dict.items():
                flag = 0
                for category, keywords in categories.items():
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        if any(keyword_lower == phrase
                               for phrase in possible_phrases_lower):
                            flag += 1
                            new_row = {
                                "domain": domain,
                                "category": category,
                                "texts": sentence,
                                "prompts": prompt
                            }
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
    sampled_gender = balanced_gender.sample(frac=0.2)

    # Step 4: Select non-'gender' domain data
    non_gender_data = df[df['domain'] != 'gender']

    # Step 5: Combine the datasets
    final_df = pd.concat([sampled_gender, non_gender_data])
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description='Process text to extract prompts based on categories.')
    parser.add_argument(
        '--file_name',
        type=str,
        help='Path to the YAML file containing categories dictionary')
    parser.add_argument('--sentence_threshold',
                        type=int,
                        default=20,
                        help='Sentence length threshold')
    parser.add_argument('--prompt_length',
                        type=int,
                        default=8,
                        help='Prompt length in terms of number of words')
    parser.add_argument('--num_processes',
                        type=int,
                        default=16,
                        help='Number of processes to use')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gender_dict = load_categories(script_dir / "gender.yaml")
    profession_dict = load_categories(script_dir / "profession.yaml")

    categories_dict = {'gender': gender_dict, 'profession': profession_dict}
    dataset = load_dataset("SetFit/toxic_conversations")['train']
    data = dataset.to_pandas()
    data = data[data['label_text'] == 'toxic']

    # Split data into chunks for each process
    data_chunks = np.array_split(data['text'], args.num_processes)

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    processes = []
    for i, data_chunk in enumerate(data_chunks):
        p = multiprocessing.Process(target=worker,
                                    args=(i, data_chunk, categories_dict,
                                          args.sentence_threshold,
                                          args.prompt_length, results_queue))
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
    print(results_df.groupby('domain').size())

    # Save results directly
    results_df.to_json(script_dir / "jigsaw.json")

    # Save results to JSON files by domain
    # for domain, group_df in results_df.groupby('domain'):
    #     file_path = f"{domain}_result.json"
    #     group_df.to_dict(orient='records')
    #     with open(file_path, 'w', encoding='utf-8') as json_file:
    #         json.dump(group_df.to_dict(orient='records'),
    #                   json_file,
    #                   ensure_ascii=False,
    #                   indent=4)
    #         print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()
