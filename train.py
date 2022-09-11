import argparse
from pathlib import Path

from tinkoff_ngram.ngram import NGramConstructor, NGramManager

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    help="path to the directory with texts for learning",
    required=True,
)
parser.add_argument(
    "--model",
    help="path to the file where model will be saved",
    required=True,
)
parser.add_argument(
    "--ngram-size",
    help="length of the NGram (N)",
    default=3,
    required=False,
)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    model_file = Path(args.model)
    ngram_size = int(args.ngram_size)

    assert data_dir.exists(), "The specified input directory does not exist"
    assert data_dir.is_dir(), "The specified input directory is not a directory"

    constructor = NGramConstructor(data_dir, ngram_size)
    manager = NGramManager(ngrams=constructor.ngrams)

    print(f"Saving to {model_file.absolute()}")
    manager.save_to_file(model_file)
