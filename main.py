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

torch.cuda.manual_seed_all(1)
np.random.seed(1)


def calculate_classification_accuracy(predict, target):
    predict_label = torch.argmax(predict.data, 1)
    correct_pred = (predict_label == target.data).sum().item()
    return correct_pred


def train(args):
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
                             shuffle=True,
                             drop_last=True)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True)

    G = AlexNet(num_classes=args.label_dim, Kbits=args.Kbits)
    G = G.cuda().float()
    state_dict = load_preweights(G, args.initialized)
    G.load_state_dict(state_dict)

    crossentropy = nn.CrossEntropyLoss()
    optimizer_G = torch.optim.Adam(G.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)

    # Adversarial ground truths
    valid = Variable(torch.Tensor(args.batchsize, 1).fill_(1.0),
                     requires_grad=False).cuda()
    fake = Variable(torch.Tensor(args.batchsize, 1).fill_(0.0),
                    requires_grad=False).cuda()

    D = Uniform_D(Kbits=args.Kbits)
    D = D.cuda().float()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    adversarial_loss = torch.nn.BCELoss()

    # hashing loss
    contrast = Contrast_Loss(margin=args.margin)
    quantization = Quantization_Loss()

    best_value = 0
    for epoch in range(1, args.EPOCH + 1):
        G.train()
        D.train()
        step = 0
        description = "training :" + str(epoch) + "/" + str(args.EPOCH)
        with tqdm(trainloader, desc=description) as iterator:
            for i, (data, target) in enumerate(iterator):
                data, target = data.cuda(), target.cuda()
                gap_softmax, softmax, hash_codes = G(data)

                # ----------------------- calculate the total loss for hash learning ---------------------------
                loss_images_softmax = crossentropy(softmax, target)
                loss_gap_softmax = crossentropy(gap_softmax, target)
                quan = quantization(hash_codes)
                hinge = contrast(hash_codes, target)
                hash_loss = hinge + args.alpha * loss_images_softmax + args.gamma * loss_gap_softmax + args.theta * quan

                # ------------------------------- generate and adversarial stage -------------------------------
                # ------------------------------- training the generate
                g_loss = 0
                if step % 5 == 0:
                    g_loss = adversarial_loss(D(hash_codes), valid)

                loss = hash_loss + g_loss
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()

                # ------------------------------- training the discriminator
                if epoch < 2:
                    real_binary_data = generate_binary_distribution(
                        data.size(0), dim=args.Kbits)
                    real_binary = Variable(
                        torch.from_numpy(real_binary_data).type(
                            torch.FloatTensor),
                        requires_grad=False).cuda()
                    real_loss = adversarial_loss(D(real_binary), valid)
                    fake_loss = adversarial_loss(D(hash_codes.detach()), fake)
                    d_loss = real_loss + fake_loss

                    optimizer_D.zero_grad()
                    d_loss.backward()
                    optimizer_D.step()

                # ------------------------ displaying the loss value during training ---------------------------
                hash_softmax_correct = calculate_classification_accuracy(
                    softmax, target) / target.size(0)
                cam_softmax_correct = calculate_classification_accuracy(
                    gap_softmax, target) / target.size(0)
                information = "Loss: {:.4f}, Hash codes classification {:.2f}, cam classification {:.2f}".format(
                    loss.item(), hash_softmax_correct, cam_softmax_correct)
                iterator.set_postfix_str(information)

        if epoch % 10 == 0:
            G.eval()
            description = "testing"
            accumulated_gap_classification, accumulated_hash_classification, accumulated_number = 0, 0, 0
            with tqdm(testloader, desc=description) as iterator:
                for i, (data, target) in enumerate(iterator):
                    data, target = data.cuda(), target.cuda()
                    gap_softmax, softmax, hash_codes = G(data)

                    # ------------------------ displaying the classification accuracy in testing data ---------------------------
                    accumulated_hash_classification += calculate_classification_accuracy(
                        softmax, target)
                    accumulated_gap_classification += calculate_classification_accuracy(
                        gap_softmax, target)
                    accumulated_number = accumulated_number + data.size(0)

                    information = "Testing, Hash codes classification {:.2f}, cam classification {:.2f}".format(
                        accumulated_hash_classification / accumulated_number,
                        accumulated_gap_classification / accumulated_number)
                    iterator.set_postfix_str(information)

            # whether save the model parameters
            if accumulated_hash_classification > best_value:
                best_value = accumulated_hash_classification
                if os.path.exists(args.parameters) is False:
                    os.makedirs(args.parameters)
                    print("saved parameters path is  : " +
                          str(args.parameters))
                torch.save(G.state_dict(), args.parameters + "/G.ckpt")
                torch.save(D.state_dict(), args.parameters + "/D.ckpt")
                print(
                    "**********************saved trained weights***************************"
                )
