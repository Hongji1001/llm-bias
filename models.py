import torch
import transformers
from functools import partial
import logging

class GPT2ForSequenceClassification(transformers.PreTrainedModel):

    def __init__(self, model_name_or_path, num_labels):
        super().__init__(transformers.GPT2Config.from_pretrained(model_name_or_path))
        self.gpt2 = transformers.GPT2Model.from_pretrained(model_name_or_path)
        self.classifier = torch.nn.Linear(self.gpt2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # Using the hidden state of the last token
        hidden_state = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(hidden_state)
        return logits


class BertForSequenceClassification:
    def __new__(self, model_name_or_path, num_labels):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        return model
    

class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, num_labels):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, num_labels):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        return model


class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)


class LlamaForCausalLM:
    def __new__(self, model_name_or_path):
        return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)
   
    
class XLNetLMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.XLNetLMHeadModel.from_pretrained(model_name_or_path)
    
    
class LlamaGuard:
    def __new__(self):
        return transformers.AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b")
    

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
        
class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        logging.info("Using INLPGPT2LMHeadModel")
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(cls, model_name_or_path, projection_matrix):
        super().__init__(cls, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(cls.func)
        logging.info("Using INLPGPT2LMHeadModel")
        return model

class INLPXLNetLMHeadModel(_INLPModel):
    def __new__(cls, model_name_or_path, projection_matrix):
        super().__init__(cls, model_name_or_path, projection_matrix)
        model = transformers.XLNetLMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(cls.func)
        logging.info("Using INLPXLNetLMHeadModel")
        return model

class INLPLlama2LMHeadModel(_INLPModel):
    def __new__(cls, model_name_or_path, projection_matrix):
        super().__init__(cls, model_name_or_path, projection_matrix)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)
        model.model.register_forward_hook(cls.func)
        logging.info("Using INLPLlama2LMHeadModel")
        return model

class INLPOpt1_3LMHeadModel(_INLPModel):
    def __new__(cls, model_name_or_path, projection_matrix):
        super().__init__(cls, model_name_or_path, projection_matrix)
        model = transformers.OPTForCausalLM.from_pretrained(model_name_or_path)
        model.model.register_forward_hook(cls.func)
        logging.info("Using INLPOpt1_3LMHeadModel")
        return model

class INLPCTRLLMHeadModel(_INLPModel):
    def __new__(cls, model_name_or_path, projection_matrix):
        super().__init__(cls, model_name_or_path, projection_matrix)
        model = transformers.CTRLLMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(cls.func)
        logging.info("Using INLPCTRLLMHeadModel")
        return model