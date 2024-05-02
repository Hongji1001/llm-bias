# from dataset import data_loader
# from test_dataset import load_categories
import pandas as pd
from tqdm import tqdm


# augment dataset by flipping the words in sentence
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

if __name__ == '__main__':
    # "imdb_2k.json", "realtoxic_2k.json", "jigsaw_2k.json", "wikitext_2k.json", "wikitoxic_2k.json"
    for filename in ["bold"]:
        data = data_loader(filename)
        gender = load_categories("../words/gender.yaml")
        dataset = swap_gender_terms(data, gender)
        # main("gpt2", filename, "train", "CDA", None, data_raw = dataset)
        # main("xlnet", filename, "train", "CDA", None, data_raw = dataset)