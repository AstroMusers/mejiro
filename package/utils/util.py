import os


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


# TODO finish
def scientific_notation_string(float):
    # convert to Python scientific notion
    str = '{:e}'.format(float)
    num, exponent = str.split('e')

    # handle exponent
    if exponent[0] == '+':
        _, power = exponent.split('+')
        power = str(int(power))
        exponent = '$10^{' + power + '}'

    return ''.join(num, '$\cross$', exponent)
