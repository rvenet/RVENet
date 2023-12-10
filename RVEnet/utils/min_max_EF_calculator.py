import json
import sys

def calculate_min_max_EF(path: str):
    with open(path, "r") as data:
        data = json.load(data)

    max_EF = float("-inf")
    min_EF = float("inf")
    for patient in data:
        EF = float(data[patient]['EF'])
        if  EF> max_EF:
            max_EF = EF
        if EF < min_EF:
            min_EF = EF

    print('min EF: {}, max EF: {}'.format(min_EF, max_EF))
    return min_EF, max_EF


if __name__ == '__main__':
    json_path = sys.argv[1]
    calculate_min_max_EF(json_path)