# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from model.bert.bert_model import BERT_FOR_POI, ModelOutput

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.l2 = nn.Linear(hidden_dim, in_dim)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class SiamBERT(BERT_FOR_POI):

    def __init__(self, config):
        super().__init__(config)

        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.predictor = prediction_MLP(in_dim = config.hidden_size, hidden_dim = int(config.hidden_size / 2))

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
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        loss = self.compute_mlm_loss(outputs.mlm_head_logits, batch['labels'])
        self.log("val_loss", loss, prog_bar=True)