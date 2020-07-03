import numpy as np
from map import cal_mAP


def readtxt(path):
    label = []
    with open(path, 'r') as f:
        x = f.readlines()
        for name in x:
            temp = int(name.strip().split()[1])
            label.append(temp)
    label = np.array(label)
    return label


root_path = "/home/admin1/PytorchProject/FAH/Behaviour/UC_Behaviour/"
txtpath = "/home/admin1/PytorchProject/FAH/dataset/UC_Merced/"
train_label = readtxt(txtpath + "/train.txt")
test_label = readtxt(txtpath + "/test.txt")
label = np.concatenate([train_label, test_label], axis=0)

traindata = np.load(root_path + "Net_Alexnet_Orig_codes/trainfeatures.npy")
testdata = np.load(root_path + "Net_Alexnet_Orig_codes/testfeatures.npy")
data = np.concatenate([traindata, testdata], axis=0)
database = [data, label]
scores = cal_mAP(database, database)
