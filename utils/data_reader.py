import typing
import json

import numpy as np
import tqdm


def tokens_to_indexes(embedding_dict: typing.Dict, embedding_dim: int, reserved: typing.List[str]) -> typing.Dict:
    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), len(reserved))}
    for reserved_idx in range(len(reserved)):
        token2idx[reserved[reserved_idx]] = reserved_idx
        embedding_dict[reserved[reserved_idx]] = np.zeros(embedding_dim, dtype=np.float32)

    return token2idx


def load_words_embeddings(filepath: str, embedding_dim: int, words_set: typing.Set,
                          reserved: typing.List[str] = None) -> typing.Tuple[typing.List, typing.Dict]:
    if reserved is None:
        reserved = ["PAD", "OOV"]

    embedding_dict = {}
    print("Loading embeddings from ", filepath, "...")
    with open(filepath, "r", encoding='utf-8') as pretrained_embeddings:
        for pretrained_embedding in tqdm.tqdm(pretrained_embeddings):
            word = pretrained_embedding.split()[0]
            embedding_vector = np.asarray(pretrained_embedding.split()[1:], dtype=np.float32)
            if word in words_set:
                embedding_dict[word] = embedding_vector

    token2idx = tokens_to_indexes(embedding_dict, embedding_dim, reserved)
    idx2emb = {idx: embedding_dict[token] for token, idx in token2idx.items()}

    embedding_mat = [idx2emb[idx] for idx in range(len(idx2emb))]

    return embedding_mat, token2idx


def read_preprocessed_dataset(filepath: str) -> typing.List[typing.Dict]:
    preprocessed_dataset = []
    print("Reading preprocessed dataset from ", filepath, "...")
    with open(filepath, "r", encoding="utf-8") as preprocessed_dataset_file:
        for sample in tqdm.tqdm(preprocessed_dataset_file):
            preprocessed_dataset.append(json.loads(sample))

    return preprocessed_dataset


def get_words_set(preprocessed_dataset: typing.List[typing.Dict]):
    words_set = set()
    for sample in preprocessed_dataset:
        for token in sample["question"] + sample["context"]:
            if token not in words_set:
                words_set.add(token)
                words_set.add(token.lower())
    return words_set
