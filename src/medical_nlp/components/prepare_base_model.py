import os
import torch
from torchsummary import summary
from medical_nlp import logger
import transformers
from torch import nn
from pathlib import Path
from medical_nlp.entity.config_entity import PrepareBaseModelConfig


class BERT(nn.Module):
    def __init__(self, num_classes):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(self.bert_model.pooler.dense.in_features, num_classes)
        
    def forward(self,ids,mask=None,token_type_ids=None):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.out(o2)
        
        return out



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_base_model(self):
        base_bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.save_model(checkpoint=base_bert_model, path=self.config.nlp_base_model_path)
        nlp_model = BERT(self.config.params_classes)
        nlp_model.to(self.device)
        return nlp_model
    
    @staticmethod
    def _prepare_full_model(model, freeze_till, freeze_all=False):
        if freeze_all:
            for param in model.bert_model.parameters():
                param.requires_grad = False
        
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in model.bert_model.parameters()[:-freeze_till]:
                param.requires_grad = False
        
        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.get_base_model(),
            freeze_all=True,
            freeze_till=None
        )
        
        summary(self.full_model)
        self.save_model(checkpoint=self.full_model, path=self.config.nlp_updated_base_model_path)
        logger.info(f"saved updated model to {str(self.config.root_dir)}")

    
    @staticmethod
    def save_model(checkpoint: dict, path: Path):
        torch.save(checkpoint, path)