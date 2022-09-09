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
                sentence.lower(),
            )
        ).strip()

    def __len__(self) -> int:
        return len(self.words)
