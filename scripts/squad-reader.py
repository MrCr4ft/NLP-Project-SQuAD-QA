import typing
import json

import click


def load_raw_dataset(dataset_json):
    raw_data = json.loads(dataset_json)['data']

    dataset = {
        'questions_ids':    [],
        'questions':        [],
        'answers':          [],
        'contexts':       [],
        'contexts_ids':   []
    }

    current_context_id = 0
    for document in raw_data:
        paragraphs = document['paragraphs']
        for paragraph in paragraphs:
            dataset['contexts'].append(paragraph['context'])
            current_context_id += 1
            for question_answer in paragraphs['qas']:
                dataset['questions_ids'].append(question_answer['id'])
                dataset['questions'].append(question_answer['question'])
                dataset['contexts_ids'].append(current_context_id - 1)
                if 'answers' in question_answer:
                    dataset['answers'].append(question_answer['answers'])

    return dataset


@click.command()
@click.option("--squad-filepath", default="dataset/training_set.json", help="Filepath of the SQuAD dataset")
@click.option("--output-filepath", default="dataset/preprocessed_training_set.txt",
              help="Filepath of the preprocessed dataset")
def run(squad_filepath: str, output_filepath: str):
    with open(squad_filepath, "r") as dataset_fd:
        dataset_json = dataset_fd.read()

    dataset = load_raw_dataset(dataset_json)


run()
