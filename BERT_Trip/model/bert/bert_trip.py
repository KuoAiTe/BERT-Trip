import torch
import lightning as L
from torch import nn
from model.bert.bert_siam import SiamBERT
from transformers import AutoConfig, AutoModel
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import math
import pandas as pd
"""
class TripBERT(SiamBERT):

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
"""