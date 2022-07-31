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


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + config.num_extra_tokens, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "relative_key_query")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + config.num_extra_tokens).expand((1, -1)))

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        time_ids=None,
        aug_time_ids=None,
        position_ids = None,
        past_key_values_length = 0,
        inputs_embeds = None
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        inputs_embeds = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertTemporalModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.init_weights()

class TemporalBERT(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.config = config
        self.criterion = nn.CosineSimilarity(dim=1).cuda()
        self.bert = BertTemporalModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids = None,
        labels=None,
        attention_mask=None,
        aug_input_ids = None,
        aug_labels= None,
        aug_attention_mask=None,
        token_type_ids=None,
        time_ids=None,
        aug_time_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ghash_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_index = None,
        traj_ids = None,
        timestamps=None,
        aug_timestamps=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        z1_representation = self.bert(
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
        )[0]

        head1 = self.cls(z1_representation)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(head1.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            loss = None

        return MaskedLMOutput(
            loss=loss,
            logits=head1,
        )
