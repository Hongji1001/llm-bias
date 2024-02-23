import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer
from transformers import DistilBertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Add a classification header
class GPT2ForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.gpt2.config.hidden_size, num_labels)

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

def load_model(name="bert"):
    if name == "bert":
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif name == "roberta":
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif name == "albert":
        model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif name == "gpt-2":
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif name == "bart":
        model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    return model, tokenizer


def load_model_Classification(name="bert", num_labels=2):
    if name == 'gpt2':
        model_config = GPT2Config.from_pretrained("gpt2", num_labels=2)
        model = GPT2ForSequenceClassification("gpt2", model_config.num_labels)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        print('#'*100, tokenizer.pad_token)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dict[name], num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    return model, tokenizer


def load_model_sequence_pretrain(path, name="bert"):
    if name == 'gpt2':
        model_config = GPT2Config.from_pretrained(path)
        model = GPT2ForSequenceClassification("gpt2", model_config.num_labels)
        model.load_state_dict(torch.load(path))
        tokenizer = GPT2Tokenizer.from_pretrained(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    return model, tokenizer
