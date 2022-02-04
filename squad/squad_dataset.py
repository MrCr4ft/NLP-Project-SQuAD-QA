import typing

import torch
from torch.utils.data import Dataset
import numpy as np


class SquadDataset(Dataset):

    def __init__(self, dataset_filepath: str, include_pos: bool = False):
        self.dataset = np.load(dataset_filepath)
        self.include_pos = include_pos

        self.contexts_widxs = torch.from_numpy(self.dataset['contexts_widxs']).long()
        self.contexts_cidxs = torch.from_numpy(self.dataset['contexts_cidxs']).long()
        self.questions_widxs = torch.from_numpy(self.dataset['questions_widxs']).long()
        self.questions_cidxs = torch.from_numpy(self.dataset['questions_cidxs']).long()
        self.answers_start_location_probs = torch.from_numpy(self.dataset['answers_start_location_probs']).long()
        self.answers_end_location_probs = torch.from_numpy(self.dataset['answers_end_location_probs']).long()
        self.qids = self.dataset['qids']
        if self.include_pos:
            self.contexts_pos = torch.from_numpy(self.dataset['contexts_widxs']).long()
            self.questions_pos = torch.from_numpy(self.dataset['contexts_widxs']).long()

        self.dataset_len = len(self.contexts_widxs)

        super(SquadDataset, self).__init__()

    def __getitem__(self, sample_idx: int) -> typing.Tuple:
        item = (
            self.contexts_widxs[sample_idx],
            self.contexts_cidxs[sample_idx],
            self.questions_widxs[sample_idx],
            self.questions_cidxs[sample_idx],
            self.answers_start_location_probs[sample_idx],
            self.answers_end_location_probs[sample_idx],
            self.qids[sample_idx]
        )
        if self.include_pos:
            item += self.contexts_pos[sample_idx]
            item += self.questions_pos[sample_idx]

        return item

    def __len__(self):
        return self.dataset_len
