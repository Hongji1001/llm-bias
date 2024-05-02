import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import avgGF, gender_polarity, honest, regard, toxicity

def evaluate(dataset, outputpath):
    start_time = time.time()

    completions = dataset["completions"]
    completions_split = dataset["completions_split"]
    sensitives = dataset["sensitives"]
    # references = loaded_data["references"]
    category = dataset["category"]

    scores = {}
    # scores["bleu"] = corpus_bleu(references, completions_split)
    scores["toxicity"] = toxicity(completions, sensitives, category)
    scores["regard"] = regard(completions, sensitives, category)
    scores["honest"] = honest(completions_split, sensitives, category)
    scores["gender_polarity"] = gender_polarity(completions, sensitives, category)
    scores["avgGF"] = avgGF(completions, sensitives, category)
    # scores["guard"] = guard(completions, sensitives)

    end_time = time.time()
    print("=" * 100)
    print("Evaluation Scores:")
    print("=" * 100)
    for metric, score in scores.items():
        print(f"{metric.capitalize()} Score: {score}")
    print("=" * 100)

    grouped_scores = defaultdict(lambda: defaultdict(dict))
    for metric, domain in scores.items():
        print(metric, domain)
        for category, values in domain.items():
            for group, score in values.items():
                grouped_scores[category][group][metric] = score

    with open(outputpath, 'a') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 100 + "\n")
        
        for category, groups in grouped_scores.items():
            f.write(f"{category.capitalize()} Scores:\n")
            for group, metrics in groups.items():
                f.write(f"  {group}:\n")
                for metric, score in metrics.items():
                    f.write(f"    {metric}: {score}\n")
            f.write("-" * 100 + "\n")
        
        execution_time = end_time - start_time
        f.write(f"Function execution time: {execution_time} seconds\n")
        f.write("=" * 100 + "\n")

    print(f"Scores saved to {outputpath}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help='Path to comletions file (a jsonl file)')
    parser.add_argument('--type',
                        type=str,
                        required=False,
                        help='Path to comletions file (a jsonl file)')
    args = parser.parse_args()
    with open(args.path, 'r', encoding='utf-8') as f:
        completions = json.load(f)
    output_file_path = Path(__file__).resolve().parent / 'outputs'
    output_file_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_path / 'metrics'
    output_file_path.mkdir(parents=True, exist_ok=True)
    basename = Path(args.path).name
    filename = output_file_path / f"eval_output_{basename[:basename.find('.json')]}_{args.type}.log"
    evaluate(completions, filename)