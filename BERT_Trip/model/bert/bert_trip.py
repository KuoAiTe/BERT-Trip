
import torch
from torch import nn
from model.bert.bert_model import BERT_FOR_POI, ModelOutput

class BERT_Trip(BERT_FOR_POI):

    def __init__(self, config):
        super().__init__(config)

        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )

    def forward(self, **kwargs):
        bert_output = self.get_bert_output(**kwargs)
        hidden_states = bert_output.last_hidden_state
        mlm_head_logits = self.mlm_head(hidden_states)
        
        input_mask_expanded = kwargs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        pooled_output = sum_embeddings / sum_mask

        return ModelOutput(
            hidden_states = bert_output.last_hidden_state,
            mlm_head_logits = mlm_head_logits,
            pooled_output = pooled_output
        )
    def compute_siam_loss(self, outputs, aug_outputs):
        z1 = self.projector(outputs.pooled_output)
        p1 = self.predictor(z1)
        
        z2 = self.projector(aug_outputs.pooled_output)
        p2 = self.predictor(z2)
        siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2
        return siamese_loss
    
    def compute_loss(self, batch):
        outputs = self.forward(**batch)
        aug_outputs = self.forward(input_ids = batch['aug_input_ids'], attention_mask = batch['aug_attention_mask'])
        mlm_loss = self.compute_mlm_loss(outputs.mlm_head_logits, batch['labels']) + self.compute_mlm_loss(aug_outputs.mlm_head_logits, batch['aug_labels']) / 2
        siamese_loss = self.compute_siam_loss(outputs, aug_outputs)
        loss = mlm_loss + siamese_loss
        return loss
    