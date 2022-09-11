import argparse
from pathlib import Path

from tinkoff_ngram.data_loader import Sentence
from tinkoff_ngram.ngram import NGramManager

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prefix",
    help="Start of the sentence",
    required=False,
)
parser.add_argument(
    "--model",
    help="path to the file with model data",
    required=True,
)
parser.add_argument(
    "--length",
    help="Number of words to generate",
    required=True,
)


def pretty_print_sentence(start: list[str], cont: list[str]) -> None:
    start_s = " ".join(start)
    end_s = " ".join(cont)
    print(Sentence(start_s + " " + end_s))


if __name__ == "__main__":
    args = parser.parse_args()

    model_file = Path(args.model)
    length = int(args.length)
    # ngram_size = int(args.ngram_size)

    assert model_file.exists(), "The specified input file does not exist"
    assert model_file.is_file(), "The specified input file is not a file"

    manager = NGramManager.from_pickle(model_file)

    if args.prefix:
        prefix = Sentence(args.prefix).words
    else:
        prefix = None
    if prefix is not None:
        assert (
            len(prefix) == manager.ngram_size
        ), """
        OK I am stupid and length of prefix has to match NGram size.
        By default it's 3. Check the train's help page to see how it can be changed.
        """

        for i in range(5):
            cont = manager.generate_n_next_words(start=prefix, n=length)
            pretty_print_sentence(prefix, cont)

    else:
        for i in range(5):
            start, cont = manager.generate_with_random_start(n=length)
            pretty_print_sentence(start, cont)
