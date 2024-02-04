from transformers import BertTokenizer, BertForMaskedLM, BertForMultipleChoice, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertForMultipleChoice, AlbertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def load_model_QA(name="bert"):
    if name == "bert":
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif name == "roberta":
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif name == "albert":
        model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif name == "gpt-2":
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer


def load_model_Classification(name="bert", num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_dict[name], num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    # if name == "bert":
    #     model = BertForSequenceClassification.from_pretrained(
    #         'bert-base-uncased', num_labels=num_labels)
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # elif name == "roberta":
    #     model = RobertaForSequenceClassification.from_pretrained(
    #         'roberta-base', num_labels=num_labels)
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # elif name == "albert":
    #     model = AlbertForSequenceClassification.from_pretrained(
    #         "albert-base-v2", num_labels=num_labels)
    #     tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # elif name == "distlibert":
    #     model = DistilBertForSequenceClassification.from_pretrained(
    #         "distilbert-base-uncased", num_labels=num_labels)
    #     tokenizer = DistilBertTokenizer.from_pretrained(
    #         'distilbert-base-uncased')
    return model, tokenizer


def load_model_sequence_pretrain(path, name="bert"):
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dict[name])
    # if name == "bert":
    #     model = BertForSequenceClassification.from_pretrained(
    #         path, local_files_only=True)
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # elif name == "distlibert":
    #     model = DistilBertForSequenceClassification.from_pretrained(
    #         path, local_files_only=True)
    #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # elif name == "albert":
    #     model = AlbertForSequenceClassification.from_pretrained(
    #         path, local_files_only=True)
    #     tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # elif name == "roberta":
    #     model = RobertaForSequenceClassification.from_pretrained(
    #         path, local_files_only=True)
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return model, tokenizer
