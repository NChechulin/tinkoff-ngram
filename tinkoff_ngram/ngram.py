from dataclasses import dataclass, field
from typing import Tuple

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
