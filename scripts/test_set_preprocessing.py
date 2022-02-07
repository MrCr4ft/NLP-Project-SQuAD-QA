import typing
import json
import os

import click
import spacy
import spacy.tokens

from utils import load_raw_dataset, preprocess, get_features_from_dataset, \
    save_object, removed_useless_attributes, get_eval


@click.command()
@click.option('--test-set-filepath', default='dataset/test_set.json', help='Filepath of the SQuAD test set')
@click.option('--word2idx-filepath', default='trained_model/word2idx.json', help='Filepath of the word2idx dictionary')
@click.option('--char2idx-filepath', default='trained_model/char2idx.json', help='Filepath of the char2idx dictionary')
@click.option('--max-context-len', default=400, help='Maximum length (in terms of tokens) of a context')
@click.option('--max-query-len', default=60, help='Maximum length (in terms of tokens) of a query')
@click.option('--max-chars-per-word', default=16, help='Maximum number of chars per token')
@click.option('--compute-pos', default=False, help='Whether to let spacy compute part of speech tags or not')
@click.option('--output-dir', default='preprocessed_dataset/', help='Directory in which outputs file will be stored')
@click.option('--require-gpu', default=False, help="Whether to use the GPU at inference time (only when "
                                                   "compute-pos=True)")
def run(test_set_filepath: str, word2idx_filepath: str, char2idx_filepath: str, max_context_len: int,
        max_query_len: int, max_chars_per_word: int, compute_pos: bool, output_dir: str, require_gpu: bool):
    if compute_pos:
        print("Preprocessing will take more time to compute POS tagging\n")
    else:
        print("Preprocessing will be faster since POS tagging is not being computed\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory: %s created" % output_dir)

    test_set, _ = load_raw_dataset(test_set_filepath, 1.00)

    if require_gpu:
        spacy.require_gpu()

    word_set: typing.Set[str] = set()
    char_set: typing.Set[str] = set()

    test_set = preprocess(test_set, word_set, char_set, compute_pos)

    word2idx = json.load(open(word2idx_filepath, "r"))
    char2idx = json.load(open(char2idx_filepath, "r"))

    features = get_features_from_dataset(test_set, word2idx, char2idx, max_context_len, max_query_len,
                                         max_chars_per_word, compute_pos)

    save_object(features, os.path.join(output_dir, "test_set_features.npz"),
                True, "test set features")

    del features

    removed_useless_attributes(test_set)
    test_eval = get_eval(test_set)

    save_object(test_eval, os.path.join(output_dir, "test_set_eval.json"), False,
                "test set info for evaluation")

    del test_set


run()
