import fire
import json
from collections import Counter
import math
import numpy as np


def language_model(vocab, train, dev, lm_type="unigram"):
    with open(vocab, 'r') as file_d:
        vocab_json = json.loads(file_d.read())
    vocab_size = len(vocab_json)

    counts = Counter()
    if lm_type == "unigram":
        with open(train, 'r') as file_d:
            for line in file_d:
                for word in line.split():
                    counts[word] += 1
    else:
        pass # No counts means add-1 smoothing becomes the uniform distribution

    denom = sum(counts.values())

    entropies = []
    line_lengths = []
    with open(dev, 'r') as file_d:
        for line in file_d:
            xent = 0
            words = line.split()
            if len(words) > 100:
                continue

            for word in line.split():
                xent += -math.log((counts[word] + 1) / (denom + vocab_size))
            entropies.append(xent)
            line_lengths.append(len(words))

    return np.mean(np.array(entropies) / np.array(line_lengths))


if __name__ == "__main__":
    fire.Fire(language_model)
