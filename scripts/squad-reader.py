import click


@click.command()
@click.option("--squad-filepath", default="dataset/training_set.json", help="Filepath of the SQuAD dataset")
@click.option("--output-filepath", default="dataset/preprocessed_training_set.txt",
              help="Filepath of the preprocessed dataset")
def run(squad_filepath, output_filepath):
    pass


run()
