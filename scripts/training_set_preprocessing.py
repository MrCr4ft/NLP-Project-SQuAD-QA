import typing
import os

import click
import spacy
import spacy.tokens

from scripts.utils import load_raw_dataset, preprocess, get_features_from_dataset, \
    load_glove_embeddings, initialize_char_embeddings, save_object, get_eval


@click.command()
@click.option('--squad-filepath', default='dataset/training_set.json', help='Filepath of the SQuAD dataset')
@click.option('--training-validation-split', default=0.95, help='The percentage of the dataset used as training set')
@click.option('--glove-filepath', default='embeddings/glove.6B/glove.6B.300d.txt',
              help='Filepath of the GloVe embeddings')
@click.option('--glove-size', default=400000, help='Number of pretrained embeddings in the GloVe file')
@click.option('--glove-embedding-dim', default=300, help="Dimensionality of the GloVe embeddings")
@click.option('--char-embedding-dim', default=200, help='Dimensionality of the chars embeddings')
@click.option('--max-context-len', default=400, help='Maximum length (in terms of tokens) of a context')
@click.option('--max-query-len', default=60, help='Maximum length (in terms of tokens) of a query')
@click.option('--max-chars-per-word', default=16, help='Maximum number of chars per token')
@click.option('--compute-pos', default=False, help='Whether to let spacy compute part of speech tags or not')
@click.option('--output-dir', default='preprocessed_dataset/', help='Directory in which outputs file will be stored')
@click.option('--require-gpu', default=False, help="Whether to use the GPU at inference time (only when "
                                                   "compute-pos=True)")
def run(squad_filepath: str, training_validation_split: int, glove_filepath: str, glove_size: int,
        glove_embedding_dim: int, char_embedding_dim: int, max_context_len: int, max_query_len: int,
        max_chars_per_word: int, compute_pos: bool, output_dir: str, require_gpu: bool):

    if compute_pos:
        print("Preprocessing will take more time to compute POS tagging\n")
    else:
        print("Preprocessing will be faster since POS tagging is not being computed\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory: %s created" % output_dir)

    word_set: typing.Set[str] = set()  # will store words (i.e. tokens) in the training set to filter GloVe embeddings
    char_set: typing.Set[str] = set()  # will store chars in the training set to initialize their embeddings

    training_set, validation_set = load_raw_dataset(squad_filepath, training_validation_split)  # "linearize" SQuAD
    # simplifying its preprocessing, and split it into training and validation set.

    if require_gpu:
        spacy.require_gpu()

    training_set = preprocess(training_set, word_set, char_set, compute_pos)
    n_unique_words = len(word_set)
    n_unique_chars = len(char_set)
    print("\n%d [%d] unique words [chars] were found in the training set. All the other words [chars] will be "
          "considered OOV.\n" % (n_unique_words, n_unique_chars))

    glove_embmat, word2idx = load_glove_embeddings(glove_filepath, glove_embedding_dim, word_set, glove_size)
    char_embmat, char2idx = initialize_char_embeddings(char_set, char_embedding_dim)

    save_object({'emb_mat': glove_embmat}, os.path.join(output_dir, "glove_embeddings.npz"), True, "GloVe embeddings")
    save_object(word2idx, os.path.join(output_dir, "word2idx.json"), False, "Word to indexes dictionary")
    save_object({'emb_mat': char_embmat}, os.path.join(output_dir, "char_embeddings.npz"), True, "char embeddings")
    save_object(char2idx, os.path.join(output_dir, "char2idx.json"), False, "Char to indexes dictionary")

    training_features = get_features_from_dataset(training_set, word2idx, char2idx, max_context_len, max_query_len,
                                                  max_chars_per_word, compute_pos)
    save_object(training_features, os.path.join(output_dir, "training_set_features.npz"),
                True, "training set features")

    training_eval = get_eval(training_set)
    save_object(training_eval, os.path.join(output_dir, "training_set_eval.json"), False,
                "training set info for evaluation")

    preprocess(validation_set, word_set, char_set, compute_pos)
    new_words = len(word_set) - n_unique_words
    new_chars = len(char_set) - n_unique_chars
    print("\n%d [%d] new words [chars] were found in the validation set. These will be considered OOV\n"
          % (new_words, new_chars))

    validation_features = get_features_from_dataset(validation_set, word2idx, char2idx, max_context_len, max_query_len,
                                                    max_chars_per_word, compute_pos)
    save_object(validation_features, os.path.join(output_dir, "validation_set_features.npz"),
                True, "validation set features")

    validation_eval = get_eval(validation_set)
    save_object(validation_eval, os.path.join(output_dir, "validation_set_eval.json"), False,
                "validation set info for evaluation")


run()
