import csv

def read_file(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.read()