import copy
import typing
import json

import numpy as np
import spacy
import spacy.tokens
import tqdm


def parse_document(document: typing.Dict, current_context_idx: int,
                   dataset: typing.Dict, include_ans: bool = True) -> int:
    paragraphs = document['paragraphs']
    for paragraph in paragraphs:
        dataset['contexts'].append(paragraph['context'].replace("''", '" ').replace("``", '" '))
        current_context_idx += 1

        for question_answer in paragraph['qas']:
            dataset['questions_ids'].append(question_answer['id'])
            dataset['questions'].append(question_answer['question'].replace("''", '" ').replace("``", '" '))
            dataset['contexts_idxs'].append(current_context_idx - 1)
            if include_ans:
                dataset['answers'].append(question_answer['answers'])

    return current_context_idx


def load_raw_dataset(squad_filepath: str, train_dev_split: float,
                     include_ans: bool = True) -> typing.Tuple[typing.Dict, typing.Dict]:
    """
    Load the SQuAD dataset (version 1.1!)
    :param squad_filepath: The dataset filepath
    :param train_dev_split: The percentage of the dataset used as training set
    :param include_ans: False if we are loading a test set (with no answers)
    :return: The training set and the evaluation set as "flattened" dictionaries
    """

    assert 0.0 <= train_dev_split <= 1.00, \
        "The splitting percentage must range from 0.0 to 1.0"

    raw_dataset: typing.Dict = json.load(open(squad_filepath, "r", encoding="utf-8"))['data']

    dataset_len = len(raw_dataset)
    print("The raw dataset has %d entries" % dataset_len)
    training_set_len = int(dataset_len * train_dev_split)
    print("The first %d entries will be taken as training set" % training_set_len)

    training_set = {
        'questions_ids': [],
        'questions': [],
        'contexts': [],
        'contexts_idxs': []
    }
    if include_ans:
        training_set['answers'] = []

    validation_set = copy.deepcopy(training_set)

    current_context_idx = 0
    for training_doc in raw_dataset[:training_set_len]:
        current_context_idx = parse_document(training_doc, current_context_idx, training_set, include_ans)

    current_context_idx = 0
    for validation_doc in raw_dataset[training_set_len:]:
        current_context_idx = parse_document(validation_doc, current_context_idx, validation_set, include_ans)

    return training_set, validation_set


def find_answer_in_context(offsets, start, end):
    answer_offsets = []
    for idx, offset in enumerate(offsets):
        if not (end <= offset[0] or start >= offset[1]):
            answer_offsets.append(idx)

    assert len(answer_offsets) > 0, "Could not find answer!"

    return answer_offsets[0], answer_offsets[-1]


def get_offsets(context: str, context_tokens: typing.List[str]) -> typing.List[typing.Tuple[int, int]]:
    current = 0
    offsets = []
    for token in context_tokens:
        current = context.find(token, current)
        assert current >= 0, "Couldn't find token in text!"
        offsets.append((current, current + len(token)))
        current += len(token)

    return offsets


def get_tokens_from_nlp_doc(nlp_doc: spacy.tokens.Doc, word_set: typing.Set[str], char_set: typing.Set[str]) -> \
        typing.Tuple[typing.List[str], typing.List[typing.List[str]]]:
    """

    :param nlp_doc:
    :param word_set:
    :param char_set:
    :return:
    """
    word_tokens = []
    char_tokens = []
    for token in nlp_doc:
        token_chars = []
        if token not in word_set:
            word_set.add(token.text)
        word_tokens.append(token.text)
        for token_char in token.text:
            if token_char not in char_set:
                char_set.add(token_char)
            token_chars.append(token_char)
        char_tokens.append(token_chars)

    return word_tokens, char_tokens


def preprocess(dataset: typing.Dict, word_set: typing.Set[str], char_set: typing.Set[str],
               compute_pos: bool = False, include_ans: bool = True) -> typing.Dict:
    dataset['contexts_word_tokens'] = []
    dataset['contexts_char_tokens'] = []
    dataset['contexts_offsets'] = []
    dataset['questions_word_tokens'] = []
    dataset['questions_char_tokens'] = []
    if include_ans:
        dataset['answers_locations'] = []
    if compute_pos:
        dataset['contexts_pos'] = []
        dataset['questions_pos'] = []

    nlp = spacy.load('en_core_web_trf') if compute_pos else spacy.blank("en")

    print("Processing contexts...")
    for cidx, context in tqdm.tqdm(enumerate(dataset['contexts']), total=len(dataset['contexts'])):
        context_doc = nlp(context)
        context_word_tokens, context_char_tokens = get_tokens_from_nlp_doc(context_doc, word_set, char_set)
        context_offsets = get_offsets(context_doc.text, context_word_tokens)
        dataset['contexts_word_tokens'].append(context_word_tokens)
        dataset['contexts_char_tokens'].append(context_char_tokens)
        dataset['contexts_offsets'].append(context_offsets)
        if compute_pos:
            dataset['contexts_pos'].append([token.tag_ for token in context_doc])

    print("Processing questions...")
    for qidx, question in tqdm.tqdm(enumerate(dataset['questions']), total=len(dataset['questions'])):
        context_idx = dataset['contexts_idxs'][qidx]
        question_doc = nlp(question)
        question_word_tokens, question_char_tokens = get_tokens_from_nlp_doc(question_doc, word_set, char_set)
        dataset['questions_word_tokens'].append(question_word_tokens)
        dataset['questions_char_tokens'].append(question_char_tokens)
        if include_ans:
            answer = dataset['answers'][qidx][0]  # in the version of SQuAD used there is only one answer
            dataset['answers_locations'].append(find_answer_in_context(dataset['contexts_offsets'][context_idx],
                                                                       answer['answer_start'],
                                                                       answer['answer_start'] + len(answer['text'])))
        if compute_pos:
            dataset['questions_pos'].append([token.tag_ for token in question_doc])

    return dataset


POS_LABELS = [
    "<PAD>",  # padding added at index 0
    "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD",
    "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
    "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP",
    "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB",
    "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD",
    "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
    "WRB", "XX", "``", "_SP"
]  # see https://spacy.io/models/en#en_core_web_trf

POS2IDX = {
    pos_label: idx for idx, pos_label in enumerate(POS_LABELS)
}


def get_index_from_word(word, word2idx):
    # if we cannot find the actual word we look for variations
    for word_form in (word, word.lower(), word.capitalize(), word.upper()):
        if word_form in word2idx:
            return word2idx[word_form]
    return word2idx["<<OOV>>"]


def get_index_from_char(char, char2idx):
    if char in char2idx:
        return char2idx[char]
    return char2idx["<<OOV>>"]


def get_features_from_dataset(dataset: typing.Dict[str, typing.List], word2idx: typing.Dict[str, int],
                              char2idx: typing.Dict[str, int], max_context_len: int, max_query_len: int,
                              max_chars_per_word: int, include_pos: bool, include_ans: bool = True):
    n_examples = len(dataset['questions'])
    contexts_widxs = []
    contexts_cidxs = []
    contexts_pos = []  # filled only when compute_pos = True
    questions_widxs = []
    questions_cidxs = []
    questions_pos = []  # filled only when compute_pos = True
    answers_start_locations = []
    answers_end_locations = []
    qids = []

    print("Extracting features...")
    for qidx in tqdm.tqdm(range(n_examples), total=n_examples):
        context_widxs = np.ones(shape=max_context_len, dtype=np.int32) * word2idx["<<PAD>>"]
        context_cidxs = np.ones(shape=(max_context_len, max_chars_per_word), dtype=np.int32) * char2idx["<<PAD>>"]
        context_pos = np.zeros(shape=max_context_len, dtype=np.int32)
        question_widxs = np.ones(shape=max_query_len, dtype=np.int32) * word2idx["<<PAD>>"]
        question_cidxs = np.ones(shape=(max_query_len, max_chars_per_word), dtype=np.int32) * char2idx["<<PAD>>"]
        question_pos = np.zeros(shape=max_query_len, dtype=np.int32)

        cidx = dataset['contexts_idxs'][qidx]
        if len(dataset['contexts_word_tokens'][cidx]) > max_context_len or \
                len(dataset['questions_word_tokens'][qidx]) > max_query_len:
            continue

        for widx, word in enumerate(dataset['contexts_word_tokens'][cidx]):
            context_widxs[widx] = get_index_from_word(word, word2idx)
            for chidx, char in enumerate(dataset['contexts_char_tokens'][cidx][widx][:max_chars_per_word]):
                context_cidxs[widx, chidx] = get_index_from_char(char, char2idx)
            if include_pos:
                context_pos[widx] = POS2IDX[dataset['contexts_pos'][cidx][widx]]

        for widx, word in enumerate(dataset['questions_word_tokens'][qidx]):
            question_widxs[widx] = get_index_from_word(word, word2idx)
            for chidx, char in enumerate(dataset['questions_char_tokens'][qidx][widx][:max_chars_per_word]):
                question_cidxs[widx, chidx] = get_index_from_char(char, char2idx)
            if include_pos:
                question_pos[widx] = POS2IDX[dataset['questions_pos'][qidx][widx]]

        if include_ans:
            answer_start_location = dataset['answers_locations'][qidx][0]
            answer_end_location = dataset['answers_locations'][qidx][1]
            answers_start_locations.append(answer_start_location)
            answers_end_locations.append(answer_end_location)

        contexts_widxs.append(context_widxs)
        contexts_cidxs.append(context_cidxs)
        questions_widxs.append(question_widxs)
        questions_cidxs.append(question_cidxs)
        qids.append(dataset['questions_ids'][qidx])
        if include_pos:
            contexts_pos.append(context_pos)
            questions_pos.append(question_pos)

    output = {
        'contexts_widxs': np.array(contexts_widxs),
        'contexts_cidxs': np.array(contexts_cidxs),
        'questions_widxs': np.array(questions_widxs),
        'questions_cidxs': np.array(questions_cidxs),
        'qids': np.array(qids)
    }
    if include_ans:
        output['answers_start_locations'] = np.array(answers_start_locations)
        output['answers_end_locations'] = np.array(answers_end_locations)

    if include_pos:
        output['contexts_pos'] = contexts_pos
        output['questions_pos'] = questions_pos

    return output


def tokens_to_indexes(embedding_dict: typing.Dict, embedding_dim: int, reserved: typing.List[str]) -> typing.Dict:
    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), len(reserved))}
    for reserved_idx in range(len(reserved)):
        token2idx[reserved[reserved_idx]] = reserved_idx
        embedding_dict[reserved[reserved_idx]] = np.zeros(embedding_dim, dtype=np.float32)

    return token2idx


def load_glove_embeddings(glove_filepath: str, embedding_dim: int, words_set: typing.Set[str],
                          num_embeddings, reserved: typing.List[str] = None) -> typing.Tuple[typing.List, typing.Dict]:
    if reserved is None:
        reserved = ["<<PAD>>", "<<OOV>>"]

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
        reserved = ["<<PAD>>", "<<OOV>>"]

    embedding_mat = np.random.normal(loc=0.0, scale=0.1, size=(len(char_set) + len(reserved), embedding_dim))
    char2idx = {char: idx for idx, char in enumerate(char_set, len(reserved))}

    for reserved_idx, reserved_chr in enumerate(reserved):
        char2idx[reserved_chr] = reserved_idx
        embedding_mat[reserved_idx, :] = np.zeros(embedding_dim)

    return embedding_mat, char2idx


def save_object(obj: object, filepath: str, numpy_obj: bool = False, msg: str = "generic object"):
    print("Saving ", msg, "...")
    if not numpy_obj:
        with open(filepath, "w+") as f:
            json.dump(obj, f)
    else:
        np.savez(filepath, **obj)
    print("Saving complete")


def get_eval(dataset: typing.Dict):
    eval = {
        qid:
            {
                'context': dataset['contexts'][dataset['contexts_idxs'][qidx]],
                'question': dataset['questions'][qidx],
                'answer': dataset['answers'][qidx],
                'contexts_offsets': dataset['contexts_offsets'][dataset['contexts_idxs'][qidx]]
            }
        for qidx, qid in enumerate(dataset['questions_ids'])
    }

    return eval
