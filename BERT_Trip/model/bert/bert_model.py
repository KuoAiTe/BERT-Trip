import torch
import lightning as L
from torch import nn, Tensor
from transformers import AutoModel
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelOutput:
    hidden_states: Tensor
    mlm_head_logits: Optional[Tensor] = None
    pooled_output: Optional[Tensor] = None

class BERT_FOR_POI(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config =config
        self.bert = AutoModel.from_config(config)
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.LayerNorm(config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.vocab_size),
        )

    def get_bert_output(self, input_ids, attention_mask, **kwargs):
        return self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    
    def forward(self, **kwargs):
        bert_output = self.get_bert_output(**kwargs)
        mlm_head_logits = self.mlm_head(bert_output.last_hidden_state)
        return ModelOutput(
            hidden_states = bert_output.last_hidden_state,
            mlm_head_logits = mlm_head_logits,
        )
    def compute_loss(self, batch):
        outputs = self.forward(**batch)
        loss = self.compute_mlm_loss(outputs.mlm_head_logits, batch['labels'])
        return loss
    def compute_mlm_loss(self, prediction_scores, labels):
        masked_lm_loss = self.mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    