import numpy
import os


def test_round(round):
    numpy.savetxt(f'_test_input_{ID}', data[:round], delimiter='\t', header=f'{round}\t{column_count}\n')
    os.system(f'timeout 5 python3 get_numbers.py <_test_input_{ID} >_test_output_{ID}')
    try:
        output = numpy.loadtxt(f'_test_output_{ID}', delimiter='\t')
        assert output.shape == (2,)
        guess1, guess2 = output
        assert 0 < guess1 < 100 and 0 < guess2 < 100
    except:
        print(f'{round:4d}: FAIL!')
        return -100
    result = data[round][0]
    guesses = numpy.concatenate([[guess1, guess2], data[round][1:]])
    winner = abs(guesses - result).argmin()
    score = winner in (0, 1)
    print(f'{round:4d}: score={score:d}, result={result}, guess1={guess1}, guess2={guess2}, best={guesses[winner]}')
    return score


def main(file, test_times):
    global ID, TEST_TIMES, data, column_count, row_count

    ID = f'{file.replace("/", "_")}_{test_times}'
    TEST_TIMES = test_times

    path = f'tests/{file}.txt'
    if not os.path.exists(path):
        path = file
    data = numpy.loadtxt(path, skiprows=1)
    row_count, column_count = data.shape
    result = sum(test_round(i) for i in range(row_count))
    print('result:', result)


if __name__ == '__main__':
    from sys import argv
    main(argv[1], int(argv[2]))
