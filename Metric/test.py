import numpy as np
from mAP import cal_mAP


def readtxt(path):
    label = []
    with open(path, 'r') as f:
        x = f.readlines()
        for name in x:
            temp = int(name.strip().split()[1])
            label.append(temp)
    label = np.array(label)
    return label


train_label = readtxt(
    "/home/admin1/PytorchProject/Meta-Hash/dataset/NWPU/train.txt")
test_label = readtxt(
    "/home/admin1/PytorchProject/Meta-Hash/dataset/NWPU/test.txt")
label_data = np.concatenate([train_label, test_label], axis=0)

traincodes = np.load(
    "/home/admin1/PytorchProject/Meta-Hash/codes/trainfeatures.npy")
testcodes = np.load(
    "/home/admin1/PytorchProject/Meta-Hash/codes/testfeatures.npy")
data = np.concatenate([traincodes, testcodes], axis=0)
label = label_data

database = [data, label]
score = cal_mAP(database, database)
print(" MAP {}".format(score))
