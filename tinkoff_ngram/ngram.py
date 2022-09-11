from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from data_loader import DataLoader, Sentence
from numpy.random import choice
from tqdm import tqdm


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


@dataclass
class NGramConstructor:
    """
    Constructs a list of NGrams from data directory.
    """

    ngrams: list[NGram] = field(default_factory=list)
    __all_sentences: list[Sentence] = field(default_factory=list)

    def __init__(self, data_dir: Path, size: int) -> None:
        self.ngrams = list()
        self.__all_sentences = list()

        dl = DataLoader(data_dir)

        for file in dl.parsed_files:
            self.__all_sentences += file.sentences

        self.__construct_ngrams(size)
        hashed = self.__construct_ngrams_hash_dict()
        self.__filter_duplicate_ngrams(hashed)

    def __construct_ngrams(self, size: int) -> None:
        """
        Constructs NGrams of a given size from text files.

        Parameters
        ----------
        size : int
            _description_
        """
        print("Constructing NGrams...")

        for sent in tqdm(self.__all_sentences):
            for i in range(len(sent) - size + 1):
                words = sent.words[i : i + size]
                ng = NGram(words=words)
                if i + size < len(sent):
                    ng.add_next_word(sent, i)
                self.ngrams.append(ng)

    def __construct_ngrams_hash_dict(self) -> Dict[int, list[NGram]]:
        """
        Constructs a dict where key is hash and value is all the NGrams which
        have that hash.
        This is needed because multiple NGrams contain the same words,
        but have different `next_words`.
        Essentially, they are the same and have to be merged.

        Returns
        -------
        Dict[int, list[NGram]]
            A dictionary containing hash and all NGrams with that hash.
        """
        result: Dict[int, list[NGram]] = dict()

        for ngram in self.ngrams:
            ng_hash = hash(ngram)
            if ng_hash not in result.keys():
                result[ng_hash] = list()

            result[ng_hash].append(ngram)

        return result

    def __filter_duplicate_ngrams(self, hashed: Dict[int, list[NGram]]) -> None:
        """
        Merges duplicate ngrams into one and deletes all the repeating ones.

        Parameters
        ----------
        hashed : Dict[int, list[NGram]]
            Dict where key is hash and value is list of equivalent NGrams.
        """
        print("Cleaning up NGrams...")

        self.ngrams = list(
            map(
                lambda ngrams: self.__merge_equal_ngrams(ngrams),
                tqdm(hashed.values()),
            )
        )

    def __merge_equal_ngrams(self, ngrams: list[NGram]) -> NGram:
        """
        Takes a list of NGrams constructed from the same words and merges them.

        Parameters
        ----------
        ngrams : list[NGram]

        Returns
        -------
        NGram
            One merged NGram
        """
        for i in range(1, len(ngrams)):
            ngrams[0] += ngrams[i]
        return ngrams[0]
