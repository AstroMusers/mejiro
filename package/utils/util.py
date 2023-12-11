import os


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]
