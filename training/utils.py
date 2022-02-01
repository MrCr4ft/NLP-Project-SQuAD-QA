import typing
import re
import string
from collections import Counter

import torch


def get_answers_spans(ans_start_probs: torch.Tensor, ans_end_probs: torch.Tensor, ans_max_len: int = -1) -> \
        typing.Tuple[torch.Tensor, torch.Tensor]:
    """

    :param ans_start_probs: Softmax of the first output vector from the model (ans_start_probs[i] := probability that
    i is the starting index of the answer
    :type ans_start_probs: torch.Tensor

    :param ans_end_probs: Softmax of the second output vector from the model (ans_end_probs[i] := probability that
    i is the ending index of the answer
    :type ans_end_probs: torch.Tensor

    :param ans_max_len: Max allowed length for an answer
    :type ans_max_len: int

    :returns The most probable starting and ending indexes for the whole batch
    :rtype Tuple[torch.Tensor, torch.Tensor]
    """
    joint_probs = torch.matmul(ans_start_probs.unsqueeze(2), ans_end_probs.unsqueeze(1))  # dim 0 is the batch size
    for batch_idx in range(joint_probs.size()[0]):
        joint_probs[batch_idx] = torch.triu(joint_probs[batch_idx])  # set to 0 (i,j) entries where i > j
        if ans_max_len != -1:
            joint_probs[batch_idx] = torch.tril(joint_probs[batch_idx], ans_max_len)  # set to 0 (i,j) entries
            # where (j - i) > ans_max_len

    most_probable_start = torch.argmax(torch.max(joint_probs, dim = 1)[0], dim=1)
    most_probable_end = torch.argmax(torch.max(joint_probs, dim=2)[0], dim=1)

    return most_probable_start, most_probable_end


def get_predictions_from_spans(eval_dict: typing.Dict, qids, pred_ans_start, pred_ans_end):
    pred_dict = {}
    for qid, ans_start, ans_end in zip(qids, pred_ans_start, pred_ans_end):
        context = eval_dict[qid]["context"]
        context_offsets = eval_dict[str(qid)]["contexts_offsets"]
        start_idx = context_offsets[ans_start][0]
        end_idx = context_offsets[ans_end][1]
        pred_dict[str(qid)] = context[start_idx: end_idx]
    return pred_dict


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def compute_metrics(ground_truths, predictions):
    f1, em, total = 0, 0, len(predictions)
    for key, value in predictions.items():
        ground_truth = ground_truths[key]['answers'][0]["text"]
        prediction = value
        em += compute_em(prediction, ground_truth)
        f1 += compute_f1(prediction, ground_truth)

    return {
        'EM': 100. * em / total,
        'F1': 100. * f1 / total
    }


# Taken from the official evaluation script

def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
