import os

import requests
from tqdm import tqdm


def download_file(url, cache_dir='~/.cache', filename=None):
    if filename is None:
        filename = url.split('/')[-1]

    # download in ~/.cache
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    file_path = os.path.join(cache_dir, filename)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return file_path

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    print(f"Downloading: {filename}")
    with open(file_path, 'wb') as file, tqdm(total=total_size_in_bytes,
                                             unit='iB',
                                             unit_scale=True) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    print(f"Download completed: {file_path}")
    return file_path


if __name__ == '__main__':
    download_file(
        'https://docs.google.com/uc?export=download&id=1va3-3oBixdY4WEAOL3AvqcsGc5j2o34G',
        filename='wiki_bias/train.tsv')
