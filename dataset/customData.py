import numpy as np
import os
from PIL import Image
import torch


class MyCustomDataset():
    def __init__(self, root_path, classnum=21, transform=None):
        # stuff
        self.txtpath = root_path
        # load image path and the corresponding labels
        self.img_path = []
        self.img_label = []
        self.transform = transform

        with open(self.txtpath, 'r') as f:
            x = f.readlines()
            for name in x:
                filepath = name.strip().split()[0]
                filelabel = int(name.strip().split()[1])
                self.img_path.append(filepath)
                self.img_label.append(filelabel)

    # __getitem__() function returns the data and labels. This function is called from dataloader like this:
    def __getitem__(self, index):
        # stuff
        im_path = self.img_path[index]
        img = Image.open(im_path)
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(self.img_label[index])
        label = torch.from_numpy(label).type(torch.long)
        return img, label

    def __len__(self):
        return len(self.img_path)
