import argparse
import torch
import torchvision
import torch.nn as nn
import random

import os
from   torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import ujson
from skimage import io, transform

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method("spawn")

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='recipeqa/features', help='folder to output image features')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

try:
    os.makedirs(os.path.join(opt.outf, "train"))
except OSError:
    pass

try:
    os.makedirs(os.path.join(opt.outf, "val"))
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

class ImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_path_list = os.listdir(self.root_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        ret = {}
        img_path = self.img_path_list[idx]

        img = io.imread(os.path.join(self.root_dir, img_path))

        if self.transform:
            img = self.transform(img)

        return {
            "path": img_path,
            "img": img
        }

class RescaleToTensorAndNormalize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def process_img(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img = transform.resize(img, (self.output_size, self.output_size))
        try:
            img = img.transpose((2,0,1))
        except Exception as e:
            if img.shape==(self.output_size, self.output_size):
                print("Greyscale instead of color")
                img = np.stack((img,)*3, axis=0)
            else:
                print("Error:", img.shape)
                raise e

        img = torch.from_numpy(img)
        img = normalize(img)

        return torch.stack((*final_img_list,)).to(torch.float)

    def __call__(self, sample):
        return self.process_img(sample)

def batch_collator(device):
    
    def _internal(batch):

        paths = [ x["path"] for x in batch ]
        imgs = torch.stack((*[ x["img"] for x in batch ],)).to(device)

        final_batch = {
            "paths": paths,
            "imgs": imgs
        }

        return final_batch

    return _internal


train_dataset = ImageDataset(
    "recipeqa/images/train/images-qa",
    transform=RescaleToTensorAndNormalize(224)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=batch_collator(device=device)
)

val_dataset = ImageDataset(
    "recipeqa/images/val/images-qa",
    transform=RescaleToTensorAndNormalize(224)
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=batch_collator(device=device)
)

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()

        img_encoder = torchvision.models.vgg16_bn(pretrained=True)

        img_encoder.classifier = nn.Sequential(*list(img_encoder.classifier)[:4]) 

        self.img_encoder = img_encoder   # final size will be 4096

    def forward(self, batch):
        encoded_images = self.img_encoder(batch["imgs"])
        return encoded_images

net = Network()

def generate_features():

    for i, batch in enumerate(val_dataloader, 0):
        encoded_images = net(batch)
        for j, img_path in enumerate(batch["paths"]):
            print("encoded_images[j]:", encoded_images[j].shape)
            torch.save(encoded_images[j], os.path.join(opt.outf, "val", img_path))

    for i, batch in enumerate(train_dataloader, 0):
        encoded_images = net(batch)
        for j, img_path in enumerate(batch["paths"]):
            print("encoded_images[j]:", encoded_images[j].shape)
            torch.save(encoded_images[j], os.path.join(opt.outf, "train", img_path))

if __name__ == "__main__":
    generate_features()
