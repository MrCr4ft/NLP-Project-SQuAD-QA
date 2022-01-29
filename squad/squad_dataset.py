import typing

import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):

    def __init__(self, samples: typing.List[typing.Dict]):
        self.samples = samples
        self.dataset_size = len(self.samples)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, sample_idx: int):
        return (
            self.samples[sample_idx]["context_widxs"],
            self.samples[sample_idx]["context_cidxs"],
            self.samples[sample_idx]["question_widxs"],
            self.samples[sample_idx]["question_cidxs"],
            self.samples[sample_idx]["answer_start_location_probs"],
            self.samples[sample_idx]["answer_end_location_probs"],
            self.samples[sample_idx]["id"]
        )


def collate(data):
    cwidxs, ccidxs, qwidxs, qcidxs, ans_start, ans_end, id = zip(*data)
    cwidxs = torch.tensor(cwidxs).long()
    ccidxs = torch.tensor(ccidxs).long()
    qwidxs = torch.tensor(qwidxs).long()
    qcidxs = torch.tensor(qcidxs).long()
    ans_start = torch.from_numpy(ans_start).long()
    ans_end = torch.from_numpy(ans_end).long()

    return cwidxs, ccidxs, qwidxs, qcidxs, ans_start, ans_end
