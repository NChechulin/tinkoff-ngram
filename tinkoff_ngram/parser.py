from dataclasses import dataclass, field
from pathlib import Path

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
