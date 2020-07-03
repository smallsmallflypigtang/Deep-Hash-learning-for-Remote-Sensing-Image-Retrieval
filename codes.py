import torch
import torch.nn as nn
from models.Net import AlexNet, Uniform_D
from dataset.customData import MyCustomDataset
from loss.contrast import Contrast_Loss, Quantization_Loss
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.LoadWeights import load_preweights
from utils.generateUniformData import generate_binary_distribution
from tqdm import tqdm
import os
import argparse


def extract(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trainset = MyCustomDataset(root_path=args.img_tr, transform=transform)
    testset = MyCustomDataset(root_path=args.img_te, transform=transform)
    trainloader = DataLoader(trainset,
                             batch_size=args.batchsize,
                             shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False)

    G = AlexNet(num_classes=args.label_dim, Kbits=args.Kbits)

    G = G.cuda().float()
    G.eval()
    # parameters path
    G.load_state_dict(torch.load(args.parameters + "/G.ckpt"))
    print("sucessfully load the G parameters.")
    code_path = args.codes_dir
    print("code path is : " + str(code_path))
    if os.path.exists(code_path) is False:
        os.makedirs(code_path)

    traincodes, testcodes = [], []
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        _, _, codes = G(data)
        codes = codes.cpu().detach().numpy()
        traincodes.extend(codes)

    for batch_idx, (data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        _, _, codes = G(data)
        codes = codes.cpu().detach().numpy()
        testcodes.extend(codes)

    traincodes, testcodes = (np.array(traincodes) >
                             0.5) / 1, (np.array(testcodes) > 0.5) / 1

    # generate training codes and features
    np.save(code_path + "/traincodes.npy", traincodes)
    print("sucessfully generate train codes")

    # generate testing codes and feautures
    np.save(code_path + "/testcodes.npy", testcodes)
    print("sucessfully generate test codes")
