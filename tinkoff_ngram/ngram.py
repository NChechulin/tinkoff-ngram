from dataclasses import dataclass, field
from typing import Optional, Tuple

from data_loader import Sentence
from numpy.random import choice


@dataclass
class NGram:
    words: list[str]
    next_words: list[Tuple[str, int]] = field(default_factory=list)

    def randomly_choose_word(self) -> str:
        """
        Returns a weighted random next word.
        Weight is the number of occurrences, therefore, the more often the
        word followed current NGram, the greater the chance it will be picked.

        Returns
        -------
        str
            Random next word
        """
        words = list(map(lambda tup: tup[0], self.next_words))
        counts = list(map(lambda tup: tup[1], self.next_words))
        total = sum(counts)

        probabilities = list(map(lambda count: count / total, counts))

        return choice(
            words,
            p=probabilities,
        )  # type: ignore

    def position_in_sentence(self, sentence: Sentence) -> Optional[int]:
        """
        Returns position of the first word in the sentence or None if
        not present.

        Parameters
        ----------
        sentence : Sentence
            Sentence to look for NGram in.

        Returns
        -------
        Optional[int]
            Index of the first word or None if NGram is not present.
        """

        def __check_at_position(pos: int) -> bool:
            for i in range(pos, pos + self.size):
                if sentence.words[i] != self.words[i - pos]:
                    return False
            return True

        for i in range(len(sentence) - self.size + 1):
            if __check_at_position(i):
                return i
        return None

    @property
    def size(self) -> int:
        """
        Returns the size of an N-Gram (N)

        Returns
        -------
        int
        """
        return len(self.words)
