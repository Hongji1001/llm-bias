from pathlib import Path

import requests
from tqdm import tqdm
import pandas as pd


def download_file(url, cache_dir='~/.cache', filename=None):
    if filename is None:
        filename = url.split('/')[-1]

    # download in ~/.cache
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / filename
    if file_path.exists():
        print(f"File already exists: {file_path}")
        return file_path

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    print(f"Downloading: {filename}")
    with open(file_path, 'wb') as file, tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    print(f"Download completed: {file_path}")
    return file_path


def login_huggingface():
    from huggingface_hub import login
    path = Path.home() / '.cache' / 'huggingface' / 'token'
    try:
        with open(path, "r") as file:
            token = file.read().strip()
    except FileNotFoundError:
        print(f"Token file not found at {path}.")
    login(token)


def swap_gender_terms(df, gender_dict):
    new_df = df.copy()
    
    new_df['prompts'] = new_df['prompts'].str.lower()
    new_df['texts'] = new_df['texts'].str.lower()
    
    for index, row in tqdm(new_df.iterrows()):
        for field in ['prompts', 'texts']:
            text = row[field]
            words = text.split()
            new_words = []
            for word in words:
                if word in gender_dict['female']:
                    new_words.append(gender_dict['male'][gender_dict['female'].index(word)])
                elif word in gender_dict['male']:
                    new_words.append(gender_dict['female'][gender_dict['male'].index(word)])
                else:
                    new_words.append(word)
            new_df.at[index, field] = ' '.join(new_words)
    
    combined_df = pd.concat([df, new_df], ignore_index=True)
    print(combined_df)
    return combined_df


def get_newest_folder(directory):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None

    newest_folder = None
    latest_time = 0

    for entry in directory_path.iterdir():
        if entry.is_dir():
            creation_time = entry.stat().st_ctime
            if creation_time > latest_time:
                latest_time = creation_time
                newest_folder = entry

    return str(newest_folder) if newest_folder else None


if __name__ == '__main__':
    download_file('https://docs.google.com/uc?export=download&id=1va3-3oBixdY4WEAOL3AvqcsGc5j2o34G',
                  filename='wiki_bias/train.tsv')
