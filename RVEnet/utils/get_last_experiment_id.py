import os
import re
import sys
import numpy as np
import csv


def get_exp_id(path: str):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    folder_numbers = [re.findall("\d+", f) for f in folders]

    if len(folder_numbers) < 1:
        print("Exp{}".format(1))
        return "Exp{}".format(1)

    flattened = np.concatenate(folder_numbers)
    flattened = [int(i) for i in flattened]
    print("Exp{}".format(max(flattened) + 1))
    return "Exp{}".format(max(flattened) + 1)

if __name__ == '__main__':
    path = sys.argv[1]
    get_exp_id(path)
