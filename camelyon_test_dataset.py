import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def augmentWsi():
    test_transform = transforms.Compose([
        transforms.ToTensor(),    
    ])
    return test_transform

def getWSIData():
    test_transform = augmentWsi()
    test_dataset = WSIAugmentDataset("test", test_transform)
    return test_dataset


class WSIAugmentDataset(Dataset):
    
    def __init__(self, mode,transform):
        self.transform = transform
        pos_dir = "/mnt/e/camelyon/sample/positive"
        neg_dir = "/mnt/e/camelyon/sample/negative"
        if mode=="test":
            pos_txt = "/mnt/e/camelyon/sample/txts/test_positive_image.txt"
            neg_txt = "/mnt/e/camelyon/sample/txts/test_negative_image.txt"
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
        #random.shuffle(index)
        self.data_list = np.array(data_list)[index]
        self.label_list = np.array(label_list)[index]

    def _read_txt(self, txt_path):
        name_list = []
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            name_list.append(line)
        #random.shuffle(name_list)
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
