from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

VALID_CHARACTERS = "йцукенгшщзхъфывапролджэячсмитьбю -"


@dataclass(init=False)
class Sentence:
    """
    Represents one sentence of a text
    """

    words: list[str] = field(default_factory=list)

    def __init__(self, sentence: str) -> None:
        sentence = self.__cleanup_sentence(sentence)
        self.words = list(
            filter(
                lambda word: len(word) > 0,
                sentence.split(),
            )
        )

    def __cleanup_sentence(self, sentence: str) -> str:
        """
        Removes all unnecessary characters from a given string

        Parameters
        ----------
        sentence : str

        Returns
        -------
        str
            Cleaned-up sentence
        """
        return "".join(
            filter(
                lambda c: c in VALID_CHARACTERS,
                sentence.lower().replace("ё", "е"),
            )
        ).strip()

    def __len__(self) -> int:
        return len(self.words)

    def __hash__(self) -> int:
        return hash(" ".join(self.words))


@dataclass
class TextFile:
    """
    Opens a file, cleans up the contents and parses the text into sentences.
    """

    input_file: Path
    sentences: list[Sentence] = field(default_factory=list)

    def __post_init__(self):
        text = self.__get_text()
        self.sentences = list(
            map(lambda sentence: Sentence(sentence), self.__split_into_sentences(text))
        )

    def __get_text(self) -> str:
        """
        Reads text from the file and returns it
        """
        text = ""
        with open(self.input_file, "r") as fh:
            text = fh.read()
        return text

    def __split_into_sentences(self, text: str) -> list[str]:
        text = " ".join(text.splitlines())
        return text.split(".")


@dataclass
class DataLoader:
    """
    Loads and parses all data in a given directory.
    """

    data_dir: Path
    parsed_files: list[TextFile] = field(default_factory=list)

    def __post_init__(self):
        print("Parsing files...")

        self.parsed_files = list(
            map(
                lambda path: TextFile(input_file=path),
                tqdm(self.__get_all_file_paths()),
            )
        )

    def __get_files_recursive(self, path: Path) -> list[Path]:
        """
        Recursively traverses a given directory and saves all text files.

        Parameters
        ----------
        path : Path
            Current directory

        Returns
        -------
        list[Path]
            All text files
        """
        result: list[Path] = []
        for child in path.iterdir():
            if child.is_dir():
                result += self.__get_files_recursive(child)
            else:
                if child.suffix == ".txt":
                    result.append(child)
        return result

    def __get_all_file_paths(self) -> list[Path]:
        """
        Returns all text file paths in `data_dir`.

        Returns
        -------
        list[Path]
            Paths to text files with data inside.
        """
        return self.__get_files_recursive(self.data_dir)
