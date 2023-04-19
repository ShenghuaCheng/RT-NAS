import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from auto_augment import ImageNetPolicy, CIFAR10Policy

class Cutout(object):
    def __init__(self, length=30):
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y-self.length//2, 0, h)
        y2 = np.clip(y+self.length//2, 0, h)
        x1 = np.clip(x-self.length//2, 0, w)
        x2 = np.clip(x+self.length//2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def augmentWsi():
    train_transform = transforms.Compose([
        #transforms.Resize((64, 64)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(
        #    brightness=0.3,
        #    contrast=0.3,
        #    saturation=0.3,
        #    hue=0.2),
        #ImageNetPolicy(),
        transforms.ToTensor(),
        #Cutout(length=64),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),    
    ])
    return train_transform, eval_transform

def getWSIData():
    train_transform, eval_transform = augmentWsi()
    train_dataset = WSIAugmentDataset("train", train_transform)
    eval_dataset = WSIAugmentDataset("eval", eval_transform)
    return train_dataset, eval_dataset


class WSIAugmentDataset(Dataset):
    
    def __init__(self, mode,transform):
        self.transform = transform
        pos_dir = "/mnt/e/camelyon/sample/positive"
        neg_dir = "/mnt/e/camelyon/sample/negative"
        if mode=="train":
            pos_txt = "/mnt/e/camelyon/sample/txts/train_positive_image.txt"
            neg_txt = "/mnt/e/camelyon/sample/txts/train_negative_image.txt"
        elif mode=="eval":
            pos_txt = "/mnt/e/camelyon/sample/txts/val_positive_image.txt"
            neg_txt = "/mnt/e/camelyon/sample/txts/val_negative_image.txt"
        else:
            print("not support mode:", mode)
        pos_list = self._read_txt(pos_txt)
        pos_list = [os.path.join(pos_dir, pos_name+".jpg") for pos_name in pos_list]

        neg_list = self._read_txt(neg_txt)
        neg_list = [os.path.join(neg_dir, neg_name+".jpg") for neg_name in neg_list]
        data_list = pos_list+neg_list
        self.len = len(data_list)
        label_list = [1]*len(pos_list)+[0]*len(neg_list)
        index = [x for x in range(len(data_list))]
        random.shuffle(index)
        self.data_list = np.array(data_list)[index]
        self.label_list = np.array(label_list)[index]

    def _read_txt(self, txt_path):
        name_list = []
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            name_list.append(line)
        random.shuffle(name_list)
        return name_list

    def __getitem__(self, index):
        data_path = self.data_list[index]
        label = self.label_list[index]
        image = Image.open(data_path)
        image = image.resize((224, 224))
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len
