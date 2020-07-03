import argparse
import os
from main import train
from codes import extract
from validateMAP import validate

root = "/home/admin1/PytorchProject/FAH/"
imagenet_weights = "/home/admin1/PytorchProject/data/pytorch_weights/alexnet-owt-4df8aa71.pth"
parser = argparse.ArgumentParser(description='FAH Hashing')
parser.add_argument(
    '--phase',
    default=0,
    type=int,
    help=
    "0 means training, 1 means extract hash codes, and 2 means validate the performance of hash codes"
)
parser.add_argument('--initialized', default=imagenet_weights)
parser.add_argument('--Kbits', default=32, type=int)  # 256, 128
parser.add_argument('--margin', default=1.0, type=float)  # margin
parser.add_argument('--alpha', default=1.0, type=float)  # hash softmx
parser.add_argument('--gamma', default=1.0, type=float)  # CAM
parser.add_argument('--theta', default=0.05, type=float)  # quantization
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--gpus', default='1', type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--data_dir', default=root + '/dataset/', type=str)
parser.add_argument('--dataset', default='UC_Merced', type=str)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--parameters', default=root + '/parameters', type=str)
parser.add_argument('--codes_dir', default=root + '/codes', type=str)
args = parser.parse_args()

label_dims = {'UC_Merced': 21, 'NWPU': 45, 'AID': 30}
args.label_dim = label_dims[args.dataset]

args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_db = os.path.join(args.data_dir, args.dataset, "database.txt")
for item in vars(args):
    print("item {}, value {}".format(item, vars(args)[item]))
if args.phase == 0:
    train(args)
elif args.phase == 1:
    extract(args)
else:
    validate(args)
