import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, Kbits=32):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))

        # yapf: enable
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(nn.Dropout(), nn.Linear(4096, 4096),
                                 nn.ReLU(inplace=True))
        self.hash_layer = nn.Sequential(nn.Linear(4096, Kbits))
        self.hash_classifier = nn.Linear(Kbits, num_classes)
        self.attention = CAM(num_classes)
        self.fusion_features = Fusion_feature()
        # yapf: enable

    def forward(self, x):
        x = self.features(x)
        conv5 = self.fusion_features(x)
        gap_softmax, cam_conv5 = self.attention(conv5)

        # flatten cam_conv5
        flatten = self.maxpool(cam_conv5)
        flatten = flatten.view(flatten.size(0), -1)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        hash_code = torch.sigmoid(self.hash_layer(fc2))
        hash_softmax = self.hash_classifier(hash_code)
        return gap_softmax, hash_softmax, hash_code, fc1


class CAM(nn.Module):
    def __init__(self, num_classes):
        super(CAM, self).__init__()
        # generate dynamic classification activation map
        self.att_base_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.att_bn1 = nn.BatchNorm2d(256)
        self.att_conv1 = nn.Conv2d(256,
                                   num_classes,
                                   kernel_size=1,
                                   padding=0,
                                   bias=True)
        self.att_bn2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes,
                                   num_classes,
                                   kernel_size=1,
                                   padding=0,
                                   bias=True)
        self.att_gap = nn.AvgPool2d(13)

        self.att_conv3 = nn.Conv2d(num_classes,
                                   1,
                                   kernel_size=3,
                                   padding=1,
                                   bias=True)
        self.att_bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        ax = self.att_bn1(self.att_base_conv(x))
        ax = F.relu(self.att_bn2(self.att_conv1(ax)))
        attention_map = torch.sigmoid(self.att_bn3(self.att_conv3(ax)))
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * attention_map + x
        return ax, rx


class Fusion_feature(nn.Module):
    def __init__(self):
        super(Fusion_feature, self).__init__()
        # feature fusion
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_1x1 = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_1x1 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        conv3 = self.conv3(x)
        conv4 = self.conv4(F.relu(conv3))
        conv5 = F.relu(
            self.conv5(F.relu(conv4)) + self.conv4_1x1(conv4) +
            self.conv3_1x1(conv3))
        return conv5


class Uniform_D(nn.Module):
    def __init__(self, Kbits=32):
        super(Uniform_D, self).__init__()
        # yapf : disable
        self.dis = nn.Sequential(
            nn.Linear(Kbits, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.dis(x)
        x = torch.sigmoid(x)
        return x
