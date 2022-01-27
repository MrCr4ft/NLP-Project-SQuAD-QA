import typing
import json

import click
import spacy


def load_raw_dataset(dataset_json: str) -> typing.Dict:
    raw_data: typing.Dict = json.loads(dataset_json)['data']

    dataset = {
        'questions_ids': [],
        'questions': [],
        'answers': [],
        'contexts': [],
        'contexts_ids': []
    }

    current_context_id = 0
    for document in raw_data:
        paragraphs = document['paragraphs']
        for paragraph in paragraphs:
            dataset['contexts'].append(paragraph['context'])

            current_context_id += 1

            for question_answer in paragraph['qas']:
                dataset['questions_ids'].append(question_answer['id'])
                dataset['questions'].append(question_answer['question'])
                dataset['contexts_ids'].append(current_context_id - 1)
                dataset['answers'].append(question_answer['answers'])

    return dataset


def find_answer_in_context(offsets, start, end):
    start_token = -1
    end_token = -1
    for idx, offset in enumerate(offsets):
        if offset[0] == start:
            start_token = idx
        if offset[1] == end:
            end_token = idx

    assert start_token != -1 and end_token != -1, "Couldn't correctly find answer!"

    return start_token, end_token


def preprocess(dataset):
    output = []

    nlp = spacy.load('en_core_web_trf')
    contexts_docs = []
    print("Processing contexts...")
    for cidx, context in enumerate(dataset['contexts']):
        contexts_docs.append(nlp(context))
        percentage_completed = 100 * (cidx / len(dataset['contexts']))
        if percentage_completed.is_integer():
            print("%d%% processed" % percentage_completed)

    print("Processing questions...")
    for qidx, question in enumerate(dataset['questions']):
        question_id = dataset['questions_ids'][qidx]
        context_id = dataset['contexts_ids'][qidx]
        answer = dataset['answers'][qidx][0]  # only one answer per question
        question_doc = nlp(question)
        context_doc = contexts_docs[context_id]
        context_offsets = [(token.idx, token.idx + len(token)) for token in context_doc]
        preprocessed_entry = {
            'id': question_id,
            'question': [token.text for token in question_doc],
            'question_pos': [token.pos_ for token in question_doc],
            'question_lemma': [token.lemma_ for token in question_doc],
            'context': [token.text for token in context_doc],
            'context_pos': [token.pos_ for token in context_doc],
            'context_lemma': [token.lemma_ for token in context_doc],
            'answer': find_answer_in_context(context_offsets, answer['answer_start'],
                                             answer['answer_start'] + len(answer['text']))
        }

        output.append(json.dumps(preprocessed_entry))
        percentage_completed = 100 * (qidx / len(dataset['questions']))
        if percentage_completed.is_integer():
            print("%d%% processed" % percentage_completed)

    return output


@click.command()
@click.option('--squad-filepath', default='dataset/training_set.json', help='Filepath of the SQuAD dataset')
@click.option('--output-filepath', default='dataset/preprocessed_training_set.txt',
              help='Filepath of the preprocessed dataset')
def run(squad_filepath: str, output_filepath: str):
    with open(squad_filepath, 'r') as dataset_fd:
        dataset_json = dataset_fd.read()

    dataset = load_raw_dataset(dataset_json)
    preprocessed_entries = preprocess(dataset)

    with open(output_filepath, "w") as output_file:
        output_file.write("\n".join(preprocessed_entries))


run()
