import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
from dataset import CLMGenerationDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def gen_completions(model, tokenizer, dataset, filename='completions'):
    model.eval()
    device = 'cuda'
    model.to(device)
    batch_size = 16
    texts = dataset['prompts'].reset_index(drop=True)
    labels = dataset['texts'].reset_index(drop=True)
    sensitives = dataset['domain'].reset_index(drop=True)
    category = dataset['category'].reset_index(drop=True)
    prompt_dataset = CLMGenerationDataset(texts, labels, tokenizer)
    prompt_loader = DataLoader(prompt_dataset, batch_size)
    completions, completions_split, references = [], [], []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(prompt_loader, desc='Generating', unit='batch'):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_length=128,
                                         num_beams=5,
                                         no_repeat_ngram_size=2,
                                         early_stopping=True)

                for output in outputs:
                    completion = tokenizer.decode(output, skip_special_tokens=True)
                    completions.append(completion)
                    completions_split.append(completion.split())

                for label in labels:
                    reference = tokenizer.decode(label, skip_special_tokens=True)
                    references.append([reference.split()])
    
    data_to_save = {
        "completions": completions,
        "completions_split": completions_split,
        "sensitives": sensitives.to_list(),
        "category": category.to_list(),
        "references": references
    }

    # Construct the output directory path
    output_file_path = Path(__file__).resolve().parent / 'outputs'
    output_file_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_path / 'completions'
    output_file_path.mkdir(parents=True, exist_ok=True)

    # Construct the final path for the output file
    final_path = output_file_path / filename
    
    df = pd.DataFrame(data_to_save)
    print(f"output file save to {final_path}")
    df.to_json(final_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to process model and dataset')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        required=True,
                        help='The name or path of the model to train')
    parser.add_argument('--tokenizer_name',
                        type=str,
                        required=True,
                        help='The name of tokenizer to load')
    parser.add_argument('--prompt_file',
                        type=str,
                        required=True,
                        help='Input prompt data file (a jsonl file) '
                        'should include one column name "prompt" as training corpus')
    parser.add_argument('--filename',
                        type=str,
                        required=False,
                        help='A jsonl file to store output completions')
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    df = pd.read_json(args.prompt_file, lines=True)
    gen_completions(model, tokenizer, df, args.filename)