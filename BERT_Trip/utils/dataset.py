import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from typing import Dict

def datetime_to_interval(d):
    return int(d.hour * 6 + d.minute / 60)

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self,
    dataset_file_path: str,
    tokenizer,
    add_user_token: bool = True,
    add_time_token: bool = True,
    use_data_agumentation: bool = True,
    sep = ','
    ):
        self.add_user_token = add_user_token
        self.add_time_token = add_time_token
        self.use_data_agumentation = use_data_agumentation
        
        
        assert os.path.isfile(dataset_file_path), f"Input file path {dataset_file_path} not found"
        users = []
        trajectories = []
        aug_trajectories = []

        with open(dataset_file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            for line in lines:
                features = line.split('|')
                user_id = features[0]
                trajectory_list = np.array(features[1].split(sep))
                times = np.array([datetime.fromtimestamp(int(i)) for i in features[-1].split(sep)])
                inputs = self.aug(user_id, trajectory_list, times)
                aug_inputs = self.aug(user_id, trajectory_list, times)
                trajectories.append(inputs)
                aug_trajectories.append(aug_inputs)

        batch_encoding = tokenizer(trajectories, add_special_tokens=True, truncation=False, max_length = 9999)
        aug_batch_encoding = tokenizer(aug_trajectories, add_special_tokens=True, truncation=False, max_length = 9999)

        size = len(batch_encoding['input_ids'])
        self.examples = [{
            "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
            "aug_input_ids": torch.tensor(aug_batch_encoding['input_ids'][i], dtype=torch.long),
        } for i in range(size)
        ]

    def aug(self, user, trajectory_list, times):
        length = len(trajectory_list)
        inputs = np.array([])

        if self.add_user_token:
            inputs = np.concatenate((inputs, [user]))

        if self.use_data_agumentation:
            new_length = np.random.randint(3, length + 1)
            choices = np.sort(np.random.choice(np.arange(length), new_length, replace = False))
            trajectory = trajectory_list[choices]
        else:
            choices = np.arange(length)
            trajectory = trajectory_list

        time_chosen = times[choices]
        time_chosen = [datetime_to_interval(t) for t in time_chosen]
        time_tokens = np.array([f'time-{_}' for _ in time_chosen])

        time_tokens = [time_tokens[0], time_tokens[-1]]
        if self.add_time_token:
            inputs = np.concatenate((inputs, time_tokens))

        inputs = np.concatenate((inputs, trajectory))
        inputs = ' '.join(inputs)

        return inputs
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

