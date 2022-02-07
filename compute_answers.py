import typing
import json

import numpy as np
import click
import spacy
import spacy.tokens
import torch
from torch.nn import functional as F
import torch.utils.data
import tqdm

from scripts.utils import load_raw_dataset, preprocess, get_features_from_dataset
from model.QANet import QANet
from training.utils import get_answers_spans, get_predictions_from_spans
from squad.squad_dataset import SquadDataset


def get_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, eval_dict: typing.Dict,
                    device: str):
    model.eval()
    predictions = {}

    with torch.no_grad(), tqdm.tqdm(total=len(data_loader.dataset)) as pbar:
        for cwidxs, ccidxs, qwidxs, qcidxs, qids in data_loader:
            cwidxs = cwidxs.to(device)
            ccidxs = ccidxs.to(device)
            qwidxs = qwidxs.to(device)
            qcidxs = qcidxs.to(device)

            batch_size = cwidxs.size(0)

            logit1, logit2 = model(cwidxs, ccidxs, qwidxs, qcidxs)
            p1, p2 = F.softmax(logit1, dim=-1), F.softmax(logit2, dim=-1)

            pred_ans_start, pred_ans_end = get_answers_spans(p1, p2)

            prediction = get_predictions_from_spans(eval_dict, qids, pred_ans_start, pred_ans_end)
            predictions.update(prediction)

            pbar.update(batch_size)

    return predictions


@click.command()
@click.option('--test-set-filepath', default='dataset/test_set.json', help='Filepath of the SQuAD test set')
@click.option('--word2idx-filepath', default='trained_model/word2idx.json', help='Filepath of the word2idx dictionary')
@click.option('--char2idx-filepath', default='trained_model/char2idx.json', help='Filepath of the char2idx dictionary')
@click.option('--word-embeddings-filepath', default="trained_model/glove_embeddings.npz",
              help='Filepath of the trained word embeddings npz file')
@click.option('--char-embeddings-filepath', default="trained_model/char_embeddings.npz",
              help='Filepath of the trained char embeddings npz file')
@click.option('--max-context-len', default=400, help='Maximum length (in terms of tokens) of a context')
@click.option('--max-query-len', default=60, help='Maximum length (in terms of tokens) of a query')
@click.option('--max-chars-per-word', default=16, help='Maximum number of chars per token')
@click.option('--compute-pos', default=False, help='Whether to let spacy compute part of speech tags or not')
@click.option('--output-filepath', default='test_set_predictions.json', help='Where to store computed predictions')
@click.option('--spacy-gpu', default=False, help="Whether to use the GPU during preprocessing with Spacy")
@click.option('--model-config-filepath', default="config.json",
              help="The path of the configuration file for the model initialization")
@click.option('model-checkpoint-filepath', default="trained_model/epoch10_f1_69.59317_em_55.00389.pth.tar",
              help="The model checkpoint to load")
@click.option('--torch-gpu', default=True, help="Whether to use the GPU at inference time")
@click.option('--batch-size', default=32, help="The batch size")
def run(test_set_filepath: str, word2idx_filepath: str, char2idx_filepath: str,
        word_embeddings_filepath: str, char_embeddings_filepath: str, max_context_len: int,
        max_query_len: int, max_chars_per_word: int, compute_pos: bool, output_filepath: str,
        spacy_gpu: bool, model_config_filepath: str, model_checkpoint_filepath: str,
        torch_gpu: bool, batch_size: int):
    if compute_pos:
        print("Preprocessing will take more time to compute POS tagging\n")
    else:
        print("Preprocessing will be faster since POS tagging is not being computed\n")

    test_set, _ = load_raw_dataset(test_set_filepath, 1.00, include_ans=False)

    if spacy_gpu:
        spacy.require_gpu()

    word_set: typing.Set[str] = set()
    char_set: typing.Set[str] = set()

    test_set = preprocess(test_set, word_set, char_set, compute_pos, include_ans=False)
    del word_set
    del char_set

    word2idx = json.load(open(word2idx_filepath, "r"))
    char2idx = json.load(open(char2idx_filepath, "r"))

    test_set_features = get_features_from_dataset(test_set, word2idx, char2idx, max_context_len, max_query_len,
                                                  max_chars_per_word, compute_pos, include_ans=False)
    test_set_features = SquadDataset(load_from_disk=False, dataset_filepath="", dataset=test_set_features,
                                     include_answers=False, include_pos=False)
    del word2idx
    del char2idx

    eval_dict = {}
    for idx, qid in enumerate(test_set["questions_ids"]):
        context = test_set["contexts"][test_set["contexts_idxs"][idx]]
        context_offsets = test_set["contexts_offsets"][test_set["contexts_idxs"][idx]]
        eval_dict[qid] = {
            'context': context,
            'contexts_offsets': context_offsets
        }
    del test_set

    word_emb_mat = torch.from_numpy(np.load(word_embeddings_filepath)["emb_mat"]).double()
    char_emb_mat = torch.from_numpy(np.load(char_embeddings_filepath)["emb_mat"]).double()
    model_config = json.load(open(model_config_filepath, "r"))

    model = QANet(word_emb_mat, char_emb_mat, model_config).double()
    if torch_gpu:
        model = model.to("cuda:0")

    if torch_gpu:
        checkpoint = torch.load(model_checkpoint_filepath)
    else:
        checkpoint = torch.load(model_checkpoint_filepath, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["state_dict"])

    test_set_loader = torch.utils.data.DataLoader(
        test_set_features,
        batch_size=32,
        shuffle=False
    )

    if torch_gpu:
        predictions = get_predictions(model, test_set_loader, eval_dict, "cuda:0")
    else:
        predictions = get_predictions(model, test_set_loader, eval_dict, "cpu")

    with open(output_filepath, "w") as output:
        json.dump(predictions, output)


run()
