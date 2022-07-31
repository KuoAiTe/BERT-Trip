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
from model.bert.bert_model import BertForMaskedLM, BertModel, BertOnlyMLMHead, MaskedLMOutput

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

class SiamBERT(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.config = config
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.bert = BertModel(config, add_pooling_layer=True)
        self.projector = projection_MLP(in_dim = config.hidden_size, out_dim = config.hidden_size)
        self.predictor = prediction_MLP(in_dim = config.hidden_size, hidden_dim = int(config.hidden_size / 2))
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids=None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        time_ids=None,
        aug_time_ids=None,
        traj_ids=None,
        timestamps=None,
        aug_timestamps=None,
        pois_triplet=None,
        user_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_representation, z1_pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            time_ids=time_ids,
            aug_time_ids=aug_time_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[:2]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(z1_representation.size()).float()
        sum_embeddings = torch.sum(z1_representation * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        z1_pooled = sum_embeddings / sum_mask

        z1 = self.projector(z1_pooled)
        p1 = self.predictor(z1)
        head1 = self.cls(z1_representation)

        if aug_input_ids != None:
            z2_representation, z2_pooled = self.bert(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                position_ids=position_ids,
                time_ids=aug_time_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[:2]
            input_mask_expanded = aug_attention_mask.unsqueeze(-1).expand(z2_representation.size()).float()
            sum_embeddings = torch.sum(z2_representation * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            z2_pooled = sum_embeddings / sum_mask
            z2 = self.projector(z2_pooled)
            p2 = self.predictor(z2)
            head2 = self.cls(z2_representation)
            siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss_z1 = loss_fct(head1.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss_z2 = loss_fct(head2.view(-1, self.config.vocab_size), aug_labels.view(-1))
            mask_loss = (masked_lm_loss_z1 + masked_lm_loss_z2) / 2
            loss = mask_loss + siamese_loss
        else:
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=head1,
        )
