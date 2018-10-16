import numpy
import os


def analyze_round(round):
    result = data[round][0]
    guesses = data[round][1:]
    winner = abs(guesses - result).argmin()
    score = [0] * (column_count // 2)
    score[winner//2] += 1
    score = numpy.array(score)
    print(score)
    return score


def main(file):
    global data, column_count, row_count

    path = f'tests/{file}.txt'
    if not os.path.exists(path):
        path = file
    data = numpy.loadtxt(path, skiprows=1)
    row_count, column_count = data.shape
    result = sum((analyze_round(i) for i in range(row_count)), numpy.array([0] * (column_count // 2)))
    print('result:', result)
    print('sorted:', sorted(result, reverse=True))


if __name__ == '__main__':
    from sys import argv
    main(argv[1])
