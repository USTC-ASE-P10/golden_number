from functools import lru_cache
import numpy
import os
import sys

WEIGHT_LENGTH = 50
TEST_TIMES = 5
PULL_RATIO = 0.05
PULL_VALUE = 99


def predict(value, weight):
    if len(value) != len(weight):
        weight = weight[:len(value)]
    return value @ weight


@lru_cache(65536)
def _score(weight):
    return -sum(history[i] - predict(history[i+1:i+1+WEIGHT_LENGTH], weight) for i in range(TEST_TIMES))


def score(weight):
    return _score(tuple(weight))


def main(stdin=sys.stdin):
    global GUESS_COUNT
    global weights, data, history
    from random import random

    try:
        weights = numpy.loadtxt('weights.txt.gz')
    except OSError:
        files = os.listdir('weights')
        weights = numpy.concatenate([numpy.loadtxt(f'weights/{file}') for file in files])

    data = numpy.loadtxt(stdin, skiprows=1, ndmin=2)
    if len(data) < TEST_TIMES:
        print('10\t20\n')
        numpy.savetxt('weights.txt.gz', weights, delimiter='\t')
        return
    GUESS_COUNT = data.shape[1] - 1
    history = data[::-1, 0]

    # add or remove weights here
    ratios = [0.1, 0.3, 0.5]
    scores = numpy.fromiter(map(score, weights), float)
    wnum = weights.shape[0]
    index = numpy.argsort(scores)[int(-0.3 * wnum):]
    weights_ = weights[index]
    for weight1 in weights_:
        for weight2 in weights_:
            for ind, ratio in enumerate(ratios):
                new_weight = weight1 * ratio + weight2 * (1 - ratio)
                numpy.vstack((weights, new_weight))

    scores = numpy.fromiter(map(score, weights), float)
    pred = predict(history[:WEIGHT_LENGTH], weights[scores.argmax()])
    pred_pulled = (pred / .618 * GUESS_COUNT + (PULL_VALUE - pred)) / GUESS_COUNT * .618
    if random() < PULL_RATIO:
        print(pred_pulled, PULL_VALUE, sep='\t')
    else:
        print(pred, pred_pulled, sep='\t')

    numpy.savetxt('weights.txt.gz', weights, delimiter='\t')


if __name__ == '__main__':
    main()
