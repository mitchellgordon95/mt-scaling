import fire


def coverage(train: str, dev: str):
    """Returns the percentage of tokens in the development file that appear in the training file."""
    train_words = set()
    with open(train, 'r') as fd:
        for line in fd:
            for word in line.split():
                train_words.add(word)

    seen, total = 0, 0
    with open(dev, 'r') as fd:
        for line in fd:
            for word in line.split():
                if word in train_words:
                    seen += 1
                total += 1

    return f'{seen / total} ( {seen} / {total} )'


if __name__ == "__main__":
    fire.Fire(coverage)
