

import lightning as L
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from .data_collator import DataCollatorForLanguageModeling
from utils import LineByLineTextDataset

class MLMDataModule(L.LightningDataModule):
    def __init__(
            self,
            dataset_file_path: str,
            vocab_file: str,
            num_user_tokens: int,
            num_time_tokens: int,
            batch_size=16,\
            max_length=128,
            mlm_probability = 0.15
        ):
        super().__init__()
        self.dataset_file_path = dataset_file_path
        self.poi_tokenizer = BertTokenizer(vocab_file = vocab_file, do_lower_case = False, do_basic_tokenize = False)
        self.num_user_tokens = num_user_tokens
        self.num_time_tokens = num_time_tokens
       
        self.batch_size = batch_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    def setup(self, stage: str):
        
        dataset = LineByLineTextDataset(
            dataset_file_path = self.dataset_file_path,
            tokenizer = self.poi_tokenizer,
            add_user_token = True,
            add_time_token = True,
            use_data_agumentation = True,
        )
        self.data_collator = DataCollatorForLanguageModeling(
            dataset = dataset,
            tokenizer = self.poi_tokenizer,
            mlm = True,
            mlm_probability = self.mlm_probability,
            num_user_tokens = self.num_user_tokens,
            num_time_tokens = self.num_time_tokens,
        )
        self.train_data = dataset
        self.val_data = dataset
        self.test_data = dataset

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.data_collator)
