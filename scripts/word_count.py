import fire
from collections import Counter
import numpy as np


def word_count(train: str):
    """Counts how many times each word occurs in the training data."""
    train_words = Counter()
    with open(train, 'r') as fd:
        for line in fd:
            for word in line.split():
                train_words[word] += 1

    freqs = np.array(list(train_words.values()))

    print(f'{len(train_words)} {np.sum(freqs > 5)}')

if __name__ == "__main__":
    fire.Fire(word_count)
