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
            dataset['contexts'].append(paragraph['context'].replace("''", '" ').replace("``", '" '))

            current_context_id += 1

            for question_answer in paragraph['qas']:
                dataset['questions_ids'].append(question_answer['id'])
                dataset['questions'].append(question_answer['question'].replace("''", '" ').replace("``", '" '))
                dataset['contexts_ids'].append(current_context_id - 1)
                dataset['answers'].append(question_answer['answers'])

    return dataset


def find_answer_in_context(offsets, start, end):
    answer_offsets = []
    for idx, offset in enumerate(offsets):
        if not (end <= offset[0] or
                start >= offset[1]):
            answer_offsets.append(idx)

    assert len(answer_offsets) > 0, "Could not find answer!"

    return answer_offsets[0], answer_offsets[-1]


def preprocess(dataset):
    output = []

    # nlp = spacy.load('en_core_web_trf')
    nlp = spacy.blank("en")
    contexts_docs = []
    print("Processing contexts...")
    for cidx, context in enumerate(dataset['contexts']):
        contexts_docs.append(nlp(context))
        percentage_completed = 100 * (cidx / len(dataset['contexts']))
        if int(percentage_completed) % 10 == 0:
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
            # 'question_pos': [token.pos_ for token in question_doc],
            'question_lemma': [token.lemma_ for token in question_doc],
            'context': [token.text for token in context_doc],
            # 'context_pos': [token.pos_ for token in context_doc],
            'context_lemma': [token.lemma_ for token in context_doc],
            'answer': find_answer_in_context(context_offsets, answer['answer_start'],
                                             answer['answer_start'] + len(answer['text']))
        }

        output.append(json.dumps(preprocessed_entry))
        percentage_completed = 100 * (qidx / len(dataset['questions']))
        if int(percentage_completed) % 10 == 0:
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
