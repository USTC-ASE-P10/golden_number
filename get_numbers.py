import numpy
import os
import sys

WEIGHT_LENGTH = 50
PULL_RATIO = 0.05
PULL_VALUE = 99


def predict(value, weight):
    if len(value) != len(weight):
        weight = weight[:len(value)]
    return value @ weight


def score(weight):
    return -sum(abs(history[i] - predict(history[i+1:i+1+WEIGHT_LENGTH], weight)) for i in range(TEST_TIMES))


def main(stdin=sys.stdin, test_times=8):
    global GUESS_COUNT, TEST_TIMES
    global weights, data, history
    from random import random

    TEST_TIMES = test_times

    files = os.listdir('weights')
    weights = numpy.concatenate([numpy.loadtxt(f'weights/{file}') for file in files])

    data = numpy.loadtxt(stdin, skiprows=1, ndmin=2)
    if len(data) < TEST_TIMES:
        if not len(data):
            print('10\t20\n')
            return
        TEST_TIMES = 1
    GUESS_COUNT = data.shape[1] - 1
    history = data[::-1, 0]

    scores = numpy.fromiter(map(score, weights), float)
    # print(scores.argmax(), scores.argsort()[-5:], file=sys.stderr)
    pred = predict(history[:WEIGHT_LENGTH], weights[scores.argmax()])
    if not 0 < pred < 100:
        pred = history[0]
    pred_pulled = (pred / .618 * GUESS_COUNT + (PULL_VALUE - pred)) / GUESS_COUNT * .618
    if not 0 < pred_pulled < 100:
        pred_pulled = history[0]
    if random() < PULL_RATIO:
        print(pred_pulled, PULL_VALUE, sep='\t')
    else:
        print(pred, pred_pulled, sep='\t')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(test_times=int(sys.argv[1]))
    else:
        main()
