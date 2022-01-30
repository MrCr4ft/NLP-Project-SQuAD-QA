import typing
import json

import numpy as np
import tqdm

from squad.squad_dataset import SquadDataset

# we need these to display percentage
SQUAD_SIZE = 87599
GLOVE_6B_SIZE = 400000
GLOVE_840B_SIZE = 2196017


def tokens_to_indexes(embedding_dict: typing.Dict, embedding_dim: int, reserved: typing.List[str]) -> typing.Dict:
    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), len(reserved))}
    for reserved_idx in range(len(reserved)):
        token2idx[reserved[reserved_idx]] = reserved_idx
        # randomly initialize the unknown embedding
        embedding_dict[reserved[reserved_idx]] = np.zeros(embedding_dim, dtype=np.float32) \
            if reserved[reserved_idx] != "<PAD>" else np.random.normal(loc=0.0, scale=0.1, size=embedding_dim)

    return token2idx


def load_glove_embeddings(glove_filepath: str, embedding_dim: int, words_set: typing.Set[str],
                          reserved: typing.List[str] = None, num_embeddings: int = GLOVE_840B_SIZE) -> typing.Tuple[typing.List, typing.Dict]:
    if reserved is None:
        reserved = ["<PAD>", "<OOV>"]

    embedding_dict = {}
    print("Loading embeddings from ", glove_filepath, "...")
    with open(glove_filepath, "r", encoding='utf-8') as pretrained_embeddings:
        for pretrained_embedding in tqdm.tqdm(pretrained_embeddings, total=num_embeddings):
            word = "".join(pretrained_embedding.split()[:-embedding_dim])
            embedding_vector = np.asarray(list(map(float, pretrained_embedding.split()[-embedding_dim:])),
                                          dtype=np.float32)
            if word in words_set:
                embedding_dict[word] = embedding_vector

    token2idx = tokens_to_indexes(embedding_dict, embedding_dim, reserved)
    idx2emb = {idx: embedding_dict[token] for token, idx in token2idx.items()}

    embedding_mat = [idx2emb[idx] for idx in range(len(idx2emb))]

    return embedding_mat, token2idx


def initialize_char_embeddings(char_set: typing.Set[str], embedding_dim: int, reserved: typing.List[str] = None):
    if reserved is None:
        reserved = ["<PAD>", "<OOV>"]

    embedding_mat = np.random.normal(loc=0.0, scale=0.1, size=(len(char_set) + len(reserved), embedding_dim))
    char2idx = {char: idx for idx, char in enumerate(char_set, len(reserved))}

    for reserved_idx, reserved_chr in enumerate(reserved):
        char2idx[reserved_chr] = reserved_idx
        if reserved_chr == "<PAD>":
            embedding_mat[reserved_idx, :] = np.zeros(embedding_dim)

    return embedding_mat, char2idx


def get_words_and_char_set(preprocessed_dataset_samples: typing.List[typing.Dict], use_lemmas: bool = False):
    words_set = set()
    char_set = set()
    print("Finding unique words and characters in the dataset...")
    for sample in preprocessed_dataset_samples:
        tokens_list = sample["question"] + sample["context"] if not use_lemmas else \
            sample["question_lemma"] + sample["context_lemma"]

        for token in tokens_list:
            if token not in words_set:
                words_set.add(token)

    for word in words_set:
        char_set = char_set.union(set(word))

    return words_set, char_set


def get_index_from_word(word, word2idx):
    # if we cannot find the actual word we look for variations
    for word_form in (word, word.lower(), word.capitalize(), word.upper()):
        if word_form in word2idx:
            return word2idx[word_form]
    return word2idx["<OOV>"]


def get_index_from_char(char, char2idx):
    if char in char2idx:
        return char2idx[char]
    return char2idx["<OOV>"]


def get_features_from_sample(sample: typing.Dict, config: typing.Dict, word2idx,
                             char2idx, use_lemmas: bool = False):
    context = sample["context"] if not use_lemmas else sample["context_lemma"]
    question = sample["question"] if not use_lemmas else sample["question_lemma"]
    answer_span = sample["answer"]
    id = int(sample["id"], 16)

    context_widxs = np.ones(shape=config["context_max_len"], dtype=np.int32) * word2idx["<PAD>"]
    context_cidxs = np.ones(shape=(config["context_max_len"], config["max_chars_per_word"]), dtype=np.int32) * char2idx[
        "<PAD>"]
    question_widxs = np.ones(shape=config["question_max_len"], dtype=np.int32) * word2idx["<PAD>"]
    question_cidxs = np.ones(shape=(config["question_max_len"], config["max_chars_per_word"]), dtype=np.int32) * \
                     char2idx["<PAD>"]
    answer_start_location_probs = np.zeros(shape=config["context_max_len"], dtype=np.int32)
    answer_end_location_probs = np.zeros(shape=config["context_max_len"], dtype=np.int32)

    answer_start_location_probs[answer_span[0]] = 1
    answer_end_location_probs[answer_span[1]] = 1

    for widx, word in enumerate(context):
        context_widxs[widx] = get_index_from_word(word, word2idx)
        for cidx, char in enumerate(word[:config["max_chars_per_word"]]):
            context_cidxs[widx, cidx] = get_index_from_char(char, char2idx)

    for widx, word in enumerate(question):
        question_widxs[widx] = get_index_from_word(word, word2idx)
        for cidx, char in enumerate(word[:config["max_chars_per_word"]]):
            question_cidxs[widx, cidx] = get_index_from_char(char, char2idx)

    return \
        {
            "context_widxs": context_widxs,
            "context_cidxs": context_cidxs,
            "question_widxs": question_widxs,
            "question_cidxs": question_cidxs,
            "answer_start_location_probs": answer_start_location_probs,
            "answer_end_location_probs": answer_end_location_probs,
            "id": id
        }


def filter_sample(sample: typing.Dict, config: typing.Dict):
    return\
        len(sample["context"]) > config["context_max_len"] or \
        len(sample["question"]) > config["question_max_len"]


def read_preprocessed_dataset(filepath: str, glove_embeddings_filepath: str, word_embedding_dim: int,
                              char_embedding_dim: int, config: typing.Dict, use_lemmas: bool = False):
    preprocessed_dataset_samples = []
    print("Reading preprocessed dataset from ", filepath, "...")
    with open(filepath, "r", encoding="utf-8") as preprocessed_dataset_file:
        for sample in tqdm.tqdm(preprocessed_dataset_file, total=SQUAD_SIZE):
            preprocessed_dataset_samples.append(json.loads(sample))

    words_set, char_set = get_words_and_char_set(preprocessed_dataset_samples)
    word_embeddings, word2idx = load_glove_embeddings(glove_embeddings_filepath, word_embedding_dim, words_set)
    char_embeddings, char2idx = initialize_char_embeddings(char_set, char_embedding_dim)

    dataset = []
    print("Converting each preprocessed sample to a vector of features...")
    for sample in tqdm.tqdm(preprocessed_dataset_samples, total=len(preprocessed_dataset_samples)):
        if not filter_sample(sample, config):
            sample_features = get_features_from_sample(sample, config, word2idx, char2idx, use_lemmas)
            dataset.append(sample_features)

    dataset = SquadDataset(dataset)

    return word_embeddings, word2idx, char_embeddings, char2idx, dataset
