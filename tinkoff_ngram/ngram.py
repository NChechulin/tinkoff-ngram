from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from data_loader import Sentence
from numpy.random import choice


@dataclass
class NGram:
    words: list[str]
    next_words: Dict[str, int] = field(default_factory=dict)

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
        words = self.next_words.keys()
        counts = self.next_words.values()
        total: int = sum(counts)

        probabilities: list[float] = list(map(lambda count: count / total, counts))

        return choice(
            list(words),
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

    def add_next_word(self, sentence: Sentence, first_word_pos: int) -> None:
        next_word = sentence.words[first_word_pos + self.size]
        # Create key if does not exist
        if self.next_words.get(next_word) is None:
            self.next_words[next_word] = 0

        self.next_words[next_word] += 1

    @property
    def size(self) -> int:
        """
        Returns the size of an N-Gram (N)

        Returns
        -------
        int
        """
        return len(self.words)

    def __hash__(self) -> int:
        """
        Since we claim NGrams are equal if words are equal, we have
        to write a custom hash function which does not care about
        next_words.
        """
        return hash(" ".join(self.words))

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

    def __iadd__(self, __o: "NGram") -> "NGram":
        for next_word in __o.next_words:
            if not self.next_words.get(next_word):
                self.next_words[next_word] = 0

            self.next_words[next_word] += __o.next_words[next_word]
        return self
