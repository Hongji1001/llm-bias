import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          PreTrainedModel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT2ForSequenceClassification(PreTrainedModel):

    def __init__(self, model_name, num_labels):
        super().__init__(GPT2Config.from_pretrained(model_name))
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.gpt2.config.hidden_size,
                                          num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # Using the hidden state of the last token
        hidden_state = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden_state)
        return logits


model_dict = {
    "bert": 'bert-base-uncased',
    "roberta": 'roberta-base',
    "albert": 'albert-base-v2',
    "distlibert": 'distilbert-base-uncased'
}

model_dict_gen = {
    "bert": 'bert-base-uncased',
    "gpt2": 'gpt2',
    "llama": 'Llama-2-7b-chat-hf'
}

# def load_model(name="bert"):
#     if name == "bert":
#         model = BertForMaskedLM.from_pretrained('bert-base-uncased')
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     elif name == "roberta":
#         model = RobertaForMaskedLM.from_pretrained('roberta-base')
#         tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     elif name == "albert":
#         model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
#         tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#     elif name == "gpt-2":
#         model = GPT2LMHeadModel.from_pretrained('gpt2')
#         tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     elif name == "bart":
#         model = BartForConditionalGeneration.from_pretrained(
#             'facebook/bart-large')
#         tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
#     return model, tokenizer


def load_model_Classification(name="bert", num_labels=2):
    if name == 'gpt2':
        model_config = GPT2Config.from_pretrained("gpt2",
                                                  num_labels=num_labels)
        model = GPT2ForSequenceClassification("gpt2", model_config.num_labels)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print('#' * 100, tokenizer.pad_token)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dict[name], num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    return model, tokenizer


def load_model_Generation(name="bert"):

    model = AutoModelForCausalLM.from_pretrained(model_dict_gen[name])
    tokenizer = AutoTokenizer.from_pretrained(model_dict_gen[name])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer


def load_model_sequence_pretrain(path, name="bert"):
    if name == 'gpt2':
        model_config = GPT2Config.from_pretrained(path)
        model = GPT2ForSequenceClassification("gpt2", model_config.num_labels)
        model.load_state_dict(torch.load(path))
        tokenizer = GPT2Tokenizer.from_pretrained(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    return model, tokenizer


def llama_guard():
    from huggingface_hub import login
    path = os.path.join(os.path.expanduser('~'), '.cache')
    try:
        with open(path, "r") as file:
            token = file.read().strip()
    except FileNotFoundError:
        print(f"Token file not found at {path}.")
    login(token)
    model_id = "meta-llama/LlamaGuard-7b"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    
    return model, tokenizer
