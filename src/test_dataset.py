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


# def balance_data(data, max_samples=2000, random_state=None):
#     all_data = pd.DataFrame()
#     unique_domains = data['domain'].unique()

#     for domain in unique_domains:
#         print(domain)
#         domain_data = data[data['domain'] == domain]
#         unique_categories = domain_data['category'].unique()
#         # Determine the global minimum count within this domain, with a cap at max_samples divided by the number of categories
#         min_count = min(domain_data['category'].value_counts().min(),
#                         int(max_samples / len(unique_categories)))
#         domain_balanced = []

#         for category in unique_categories:
#             print(category)
#             category_data = domain_data[domain_data['category'] == category]
#             sample_size = min(len(category_data), min_count)
#             if len(category_data) > 0:
#                 sampled_data = category_data.sample(n=sample_size,
#                                                     random_state=random_state)
#                 domain_balanced.append(sampled_data)

#         all_data = pd.concat([all_data, pd.concat(domain_balanced)])

#     return all_data


def post_process(data: pd.DataFrame):
    # Step 2: Balance 'male' and 'female' counts
    # male_data = gender_data[gender_data['category'] == 'male']
    # female_data = gender_data[gender_data['category'] == 'female']
    # min_count = min(len(male_data), len(female_data))
    # balanced_male = male_data.sample(n=min_count)
    # balanced_female = female_data.sample(n=min_count)
    # balanced_gender = pd.concat([balanced_male, balanced_female])
    max_samples = 3000
    random_state = 42
    all_data = pd.DataFrame()
    unique_domains = data['domain'].unique()

    for domain in unique_domains:
        print(domain)
        domain_data = data[data['domain'] == domain]
        unique_categories = domain_data['category'].unique()
        # # Determine the global minimum count within this domain, with a cap at max_samples divided by the number of categories
        # min_count = min(domain_data['category'].value_counts().min(),
        #                 int(max_samples / len(unique_categories)))
        domain_balanced = []

        for category in unique_categories:
            print(category)
            category_data = domain_data[domain_data['category'] == category]
            sample_size = min(len(category_data), int(max_samples / len(unique_categories)))
            if len(category_data) > 0:
                sampled_data = category_data.sample(n=sample_size, random_state=random_state)
                domain_balanced.append(sampled_data)

        all_data = pd.concat([all_data, pd.concat(domain_balanced)])
    return all_data


def calculate_scores(df, toxicity_evaluator, regard_evaluator):
    # 转换DataFrame中的文本列为列表
    input_texts = df['texts'].tolist()

    # 使用toxicity评估器批量计算得分
    toxicity_results = toxicity_evaluator.compute(predictions=input_texts)
    toxicity_scores = toxicity_results["toxicity"]

    # 使用regard评估器批量计算得分
    regard_results = regard_evaluator.compute(data=input_texts)
    regard_scores = [d[0]['score'] for d in regard_results['regard'] for l in d if l['label'] == 'negative']

    # 将得分添加到DataFrame
    df['toxicity'] = toxicity_scores
    df['regard'] = regard_scores
    df['total_score'] = df['toxicity'] + df['regard']

    return df


def select_top_2k_per_category(df):
    # 加载评估器
    toxicity_evaluator = evaluate.load("toxicity", module_type="measurement")
    regard_evaluator = evaluate.load("regard", module_type="measurement")

    # 计算得分
    df = calculate_scores(df, toxicity_evaluator, regard_evaluator)

    domain_category_counts = df.groupby('domain')['category'].nunique().reset_index(name='unique_categories')
    domain_category_counts['per_category_quota'] = 2000 // domain_category_counts['unique_categories']
    df = df.merge(domain_category_counts[['domain', 'per_category_quota']], on='domain')

    def select_by_quota(group_df):
        quota = int(group_df['per_category_quota'].iloc[0])    # 因为每个 group 的配额相同，所以取第一个即可
        return group_df.nlargest(min(quota, len(group_df)), 'total_score')

    top_2k_per_domain = df.groupby(['domain', 'category'], group_keys=False).apply(select_by_quota)

    return top_2k_per_domain


def main():
    parser = argparse.ArgumentParser(description='Process text to extract prompts based on categories.')
    parser.add_argument('--file_name', type=str, help='Path to the YAML file containing categories dictionary')
    parser.add_argument('--sentence_threshold', type=int, default=40, help='Sentence length threshold')
    parser.add_argument('--prompt_length', type=int, default=10, help='Prompt length in terms of number of words')
    parser.add_argument('--num_processes', type=int, default=20, help='Number of processes to use')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    wordlists_dir = script_dir.parent / 'words'
    gender_dict = load_categories(wordlists_dir / "gender.yaml")
    occupation_dict = load_categories(wordlists_dir / "occupation.yaml")
    religion_dict = load_categories(wordlists_dir / "religion.yaml")
    age_dict = load_categories(wordlists_dir / "age.yaml")

    categories_dict = {'gender': gender_dict, 'occupation': occupation_dict, 'religion': religion_dict, 'age': age_dict}
    # dataset = load_dataset("SetFit/toxic_conversations")
    # data = pd.DataFrame()
    # for split in dataset.keys():
    #     df = dataset[split].to_pandas()
    #     df['split'] = split  # 添加一个列来指明当前行属于哪个数据集分区
    #     data = pd.concat([data, df], ignore_index=True)  # 合并DataFrame
    # # data = data[data['label'] == 1]
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "jigsaw_toxic_2k.json"

    # dataset = load_dataset("imdb")
    # data = pd.DataFrame()
    # for split in dataset.keys():
    #     df = dataset[split].to_pandas()
    #     df['split'] = split  # 添加一个列来指明当前行属于哪个数据集分区
    #     data = pd.concat([data, df], ignore_index=True)  # 合并DataFrame
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "imdb_2k.json"

    # dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic")
    # data = pd.DataFrame()
    # for split in dataset.keys():
    #     df = dataset[split].to_pandas()
    #     df['split'] = split    # 添加一个列来指明当前行属于哪个数据集分区
    #     data = pd.concat([data, df], ignore_index=True)    # 合并DataFrame
    # data_chunks = np.array_split(data['comment_text'], args.num_processes)
    # file_name = "wikitoxic_2k.json"

    # print()
    # dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0')['train']
    # data = dataset.to_pandas()
    # # data = data.sample(frac=0.1, random_state=42)
    # data_chunks = np.array_split(data['article'], args.num_processes)
    # file_name = "cnn_dailymail_2k.json"

    # dataset = load_dataset("wikitext", 'wikitext-103-raw-v1')['train']
    # data = dataset.to_pandas()
    # # data = data.sample(frac=0.01, random_state=42)
    # data_chunks = np.array_split(data['text'], args.num_processes)
    # file_name = "wikitext_2k.json"

    # dataset = pd.read_json(
    #     Path(__file__).resolve().parent.parent / 'data' / 'stereoset.json')
    # data_chunks = np.array_split(dataset, args.num_processes)
    # file_name = "stereoset_new.json"

    data = load_realtoxic()
    print(data)
    data_chunks = np.array_split(data, args.num_processes)
    file_name = Path(
        __file__).resolve().parent.parent / 'data' / "realtoxic_2k.json"

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
                                    args=(i, data_chunk['prompts'],
                                          data_chunk['texts'], categories_dict,
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
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())
    results_df = select_top_2k_per_category(results_df)
    # results_df = post_process(results_df)
    print(results_df.groupby(['domain']).size())
    print(results_df.groupby(['domain', 'category']).size())
    print(results_df)

    # Save results directly
    file_path = script_dir.parent / 'data' / file_name
    results_df.to_json(file_path, indent=4)
    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()
