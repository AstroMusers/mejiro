import csv
import json
from types import SimpleNamespace as Namespace


def write_json_string_to_file(json_data, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write(json_data)


def json_to_dict(json_filename):
    with open(json_filename) as jsonf:
        data = json.load(jsonf)

    jsonf.close()

    return data


def read_datasets_from_json(json_filename):
    pojo_list = []

    json_file = open(json_filename)

    data = json.load(json_file)

    for each in data:
        str_each = str(each).replace("\'", "\"")
        pojo_list.append(json.loads(str_each, object_hook=lambda d: Namespace(**d)))

    json_file.close()

    return pojo_list


def write_datasets_to_json(json_filename, dataset_list):
    json_list = []

    for dataset in dataset_list:
        json_list.append(json.dumps(dataset.__dict__))

    json_file = open(json_filename, 'w')

    json_file.write('[')

    for each in json_list:
        json_file.write(each)

        # don't write a comma after the last one
        if len(json_list) - 1 != json_list.index(each):
            json_file.write(',')

    json_file.write(']')

    json_file.close()


def write_csv_to_json(csv_filename, json_filename, uid):
    # create a dictionary
    data = {}

    # open a csv reader called DictReader
    with open(csv_filename) as csvf:
        reader = csv.DictReader(csvf)

        # convert each row into a dictionary and add it to data
        for rows in reader:
            key = rows[uid]
            data[key] = rows

    csvf.close()

    # open a json writer and use the json.dumps() function to dump data
    with open(json_filename, 'w') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

    jsonf.close()
