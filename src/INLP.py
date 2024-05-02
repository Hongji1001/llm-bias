from functools import partial

import torch
import transformers
import argparse
import os
import logging
from torch.utils.data import DataLoader
from test import test, prepare_dataset, construct_model_path
from dataset import GenerationDataset


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)

class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)
        
class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        logging.info("Using INLPGPT2Model")
        return model

class INLPXLNetLMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.XLNetLMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        logging.info("Using INLPGPT2LMHeadModel")
        return model

class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        logging.info("Using INLPGPT2LMHeadModel")
        return model

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="INLPGPT2LMHeadModel",
    choices=[
        "INLPGPT2LMHeadModel", "INLPXLNetLMHeadModel"
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM). Typically, these "
    "correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt2",
    choices=["gpt2", "xlnet/xlnet-base-cased"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    choices=["gender", "race", "religion"],
    help="What type of bias to compute the INLP projection matrix for.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved ContextDebias, CDA, or Dropout model checkpoint.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - bias_type: {args.bias_type}")

    kwargs = {}

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model_cls = globals().get(args.model)
    model = model_cls(args.load_path or args.model_name_or_path, **kwargs)

    model = model.to('cuda')
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    file_name = 'jigsaw_2k.json'
    model_path = construct_model_path('gpt2', file_name, 'INLP')
    texts, labels, sensitives, category = prepare_dataset(file_name, 'test')
    test_dataset = GenerationDataset(texts, labels, tokenizer, file_name)
    test_loader = DataLoader(test_dataset, 16)
    test(model, tokenizer, test_loader, sensitives, category, 'cuda', 'INLP')