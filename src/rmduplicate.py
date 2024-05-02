import random
from pathlib import Path
from typing import List

import evaluate
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from transformers import pipeline

# # 定义读取YAML数据的函数
# def read_yaml_from_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return yaml.safe_load(file)

# # 定义写入YAML数据到文件的函数
# def write_yaml_to_file(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         yaml.dump(data, file, allow_unicode=True)

# # 从文件读取YAML数据
# input_file_path = 'occupation.yaml'  # 替换为您的输入文件路径
# data = read_yaml_from_file(input_file_path)

# # 找出所有职位
# all_positions = [position for sublist in data.values() for position in sublist]

# # 找出重复的职位
# duplicate_positions = set([position for position in all_positions if all_positions.count(position) > 1])

# # 随机删除重复的职位，只保留一个
# for duplicate_position in duplicate_positions:
#     domains_with_duplicate = [domain for domain, positions in data.items() if duplicate_position in positions]

#     if len(domains_with_duplicate) > 1:
#         domain_to_keep = random.choice(domains_with_duplicate)
#         for domain in domains_with_duplicate:
#             if domain != domain_to_keep:
#                 data[domain].remove(duplicate_position)

# output_file_path = 'profession.yaml'  # 替换为您想要的输出文件路径
# write_yaml_to_file(data, output_file_path)



def evaluate_accuracy(completions: List[str], labels: List[str]) -> float:
    """
    Evaluate the accuracy of the toxicity model in detecting biased text.
    
    :param completions: A list of text completions.
    :param labels: A list of labels indicating whether each completion is 'biased' or 'unbiased'.
    :return: The accuracy of the toxicity model as a float.
    """
    toxicity = evaluate.load("toxicity", module_type="measurement")
    results = toxicity.compute(predictions=completions)
    predicted_biases = ['bias' if score > 0.5 else 'no bias' for score in results["toxicity"]]
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('toxicity')
    print(confusion_matrix(labels, predicted_biases))

    regard_evaluator = evaluate.load("regard", module_type="measurement")
    result = regard_evaluator.compute(data=completions)
    predicted_biases = []
    for regard_list in result['regard']:
        # Find if 'negative' has the highest score in each result entry
        highest_score_label = max(regard_list, key=lambda x: x['score'])['label']
        is_negative = 'bias' if highest_score_label == 'negative' else 'no bias'
        predicted_biases.append(is_negative)
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('regard')
    print(confusion_matrix(labels, predicted_biases))

    honest = evaluate.load('honest', 'en')
    results = []
    for completion in completions:
        result = honest.compute(predictions=[completion.split()])
        results.append(result['honest_score'])

    predicted_biases = ['bias' if score > 0.0 else 'no bias' for score in results]
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('honest')
    print(confusion_matrix(labels, predicted_biases))

    pipe = pipeline("text-classification", model="valurank/distilroberta-bias")
    results = pipe(completions)
    predicted_biases = ['bias' if result['label'] == 'BIASED' else 'no bias' for result in results]
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('valurank/distilroberta-bias')
    print(confusion_matrix(labels, predicted_biases))

    pipe = pipeline("text-classification", model="Alicewuu/bias_detection")
    results = pipe(completions)
    predicted_biases = ['bias' if result['label'] == 'BIASED' else 'no bias' for result in results]
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('Alicewuu/bias_detection')
    print(confusion_matrix(labels, predicted_biases))

    pipe = pipeline("text-classification", model="D1V1DE/bias-detection")
    results = pipe(completions)
    predicted_biases = ['bias' if result['label'] == 'BIASED' else 'no bias' for result in results]
    correct_predictions = sum(1 for true_label, predicted_label in zip(labels, predicted_biases)
                              if true_label == predicted_label)
    accuracy = correct_predictions / len(labels)
    print('D1V1DE/bias-detection')
    print(confusion_matrix(labels, predicted_biases))

    return accuracy


df = pd.read_json(Path(__file__).resolve().parent.parent / 'data' / '2000_gender_bias_data_label.jsonl', lines=True)
df['text'] = df['instruction'] + ' ' + df['output']
df['text_split'] = df['text'].apply(lambda x: x.split())
data_to_save = {
    "completions": df['text'].to_list(),
    "completions_split": df['text_split'].to_list(),
    "sensitives": ['gender'] * len(df),
    "category": df['classification'].to_list(),
}
import json
from pathlib import Path

output_file_path = Path(__file__).resolve().parent.parent / 'outputs'
output_file_path.mkdir(parents=True, exist_ok=True)
output_file_path = output_file_path / 'completions'
output_file_path.mkdir(parents=True, exist_ok=True)

# Construct the final path for the output file
final_path = output_file_path / f"gpt3.5_2000.json.json"

with open(final_path, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=True, indent=4)
print(f"output file save to {final_path}")

# print(evaluate_accuracy(df['text'].to_list(), df['classification'].to_list()))

    # pipe = pipeline("text-classification", model="D1V1DE/bias-detection")
    # pipe1 = pipeline("text-classification", model="valurank/distilroberta-bias")
    # pipe2 = pipeline("text-classification", model="Alicewuu/bias_detection")

    # result = pipe(df.head(10)['output'].to_list())
    # result1 = pipe1(df.head(10)['output'].to_list())
    # result2 = pipe2(df.head(10)['output'].to_list())

    # print(result)
    # print(result1)
    # print(result2)

    # predicted_biases2 = ['bias' if label['label'] == 'BIASED' else 'no bias' for label in result2]
    # print(predicted_biases2)
