import os
import json


# Function
def dir_check(dir_name):
    if os.path.isdir(dir_name):
        pass
    else:
        os.mkdir(dir_name)


def json_save(data, file_name):
    with open(file_name + '.json', 'w') as fp:
        json.dump(data, fp)


def json_load(file_name):
    try:
        with open(file_name) as fp:
            return json.load(fp)
    except FileNotFoundError:
        with open(file_name + '.json') as fp:
            return json.load(fp)
