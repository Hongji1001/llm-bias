from pathlib import Path
import pandas as pd
from dataset import load_bold
file_path = 'bold.json'
script_dir = Path(__file__).resolve().parent
filename = script_dir.parent / 'data' / file_path
data_raw = load_bold()
data_raw = data_raw.rename(columns={'sensitive': 'domain'})
data_raw.to_json('../data/bold.json')
print(data_raw)

# data_raw = data_raw[data_raw['sensitive'] == 'gender']

# data_raw['prompt'] = data_raw['prompts'].apply(lambda x: {'text': x})

# # Add a 'challenge' column with a placeholder value
# data_raw['challenging'] = True

# jsonl_str = data_raw.to_json(orient='records', lines=True)

# # The resulting JSON Lines formatted string
# file_path += 'l'
# file_path = script_dir.parent / 'data' / file_path
# with open(file_path, 'w') as f:
#     f.write(jsonl_str)