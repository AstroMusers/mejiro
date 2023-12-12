import os


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def scientific_notation_string(input):
    return '{:.2e}'.format(input)

# TODO finish
# def scientific_notation_string(input):
#     # convert to Python scientific notion
#     string = '{:e}'.format(input)
#     num_string, exponent = string.split('e')
#     num = str(round(float(num_string), 2))

#     # handle exponent
#     if exponent[0] == '+':
#         _, power = exponent.split('+')
#     elif exponent[0] == '-':
#         _, power = exponent.split('-')
#         power = '-' + power
        

#     power = str(int(power))
#     exponent = '10^{' + power + '}'

#     return ''.join((num, '\cross', exponent))
