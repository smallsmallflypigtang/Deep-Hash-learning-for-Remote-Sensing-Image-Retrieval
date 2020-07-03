import numpy as np
from Metric.mAP import cal_mAP
import os


def readtxt(path):
    label = []
    with open(path, 'r') as f:
        x = f.readlines()
        for name in x:
            temp = int(name.strip().split()[1])
            label.append(temp)
    label = np.array(label)
    return label


def validate(args):
    train_label = readtxt(args.img_tr)
    test_label = readtxt(args.img_te)
    label_data = np.concatenate([train_label, test_label], axis=0)

    traincodes = np.load(os.path.join(args.codes_dir, "traincodes.npy"))
    testcodes = np.load(os.path.join(args.codes_dir, "testcodes.npy"))

    data = np.concatenate([traincodes, testcodes], axis=0)
    label = label_data

    database = [data, label]
    score = cal_mAP(database, database)
    print(" MAP {}".format(score))
