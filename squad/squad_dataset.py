import typing

import torch
from torch.utils.data import Dataset
import numpy as np


class SquadDataset(Dataset):
    def __init__(self, load_from_disk: bool, dataset_filepath: str, dataset: np.ndarray, include_answers: bool = True,
                 include_pos: bool = False):
        if load_from_disk:
            dataset = np.load(dataset_filepath)

        self.include_pos = include_pos
        self.include_answers = include_answers

        self.contexts_widxs = torch.from_numpy(dataset['contexts_widxs']).long()
        self.contexts_cidxs = torch.from_numpy(dataset['contexts_cidxs']).long()
        self.questions_widxs = torch.from_numpy(dataset['questions_widxs']).long()
        self.questions_cidxs = torch.from_numpy(dataset['questions_cidxs']).long()
        self.qids = dataset['qids']
        if self.include_pos:
            self.contexts_pos = torch.from_numpy(dataset['contexts_widxs']).long()
            self.questions_pos = torch.from_numpy(dataset['contexts_widxs']).long()

        self.dataset_len = len(self.contexts_widxs)

        super(SquadDataset, self).__init__()

    def __getitem__(self, sample_idx: int) -> typing.Tuple:
        item = (
            self.contexts_widxs[sample_idx],
            self.contexts_cidxs[sample_idx],
            self.questions_widxs[sample_idx],
            self.questions_cidxs[sample_idx],
            self.qids[sample_idx]
        )
        if self.include_pos:
            item += self.contexts_pos[sample_idx]
            item += self.questions_pos[sample_idx]

        return item

    def __len__(self):
        return self.dataset_len
