import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from PIL import Image
import glob
import sklearn.metrics
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms


# Labels

with open("testset_stats", "rb") as fp:
    target_stats = pickle.load(fp)
target_stats = np.array(target_stats)
gt_label = np.zeros((target_stats.size, target_stats.max()+1))
gt_label[np.arange(target_stats.size), target_stats] = 1

# Dataset

alpha = str(1.0)


def get_image_loc(alpha='1.0', adversarial=True, data_dir="adversarial_images", length=1200):

    last_str_match = "_orig_0.png"
    if adversarial:
        last_str_match = "_ILFO_*.png"

    all_images = []
    
    for i in tqdm(range(length)):
        for j in range(2):
            img_loc = glob.glob(data_dir+"/output_"+alpha+"/img_"+str(i)+"_"+str(j)+last_str_match)[0]
            all_images.append(img_loc)
            
    return all_images


class AdversarialDataset(Dataset):
    
    def __init__(self, split='train', data_dir='adversarial_images', length=1200, split_ratio=0.8, adversarial=True, alphas="0.0,0.25,0.5,0.75,1.0"):
        
        self.split = split
        self.length = length*2
        self.split_ratio = split_ratio
        self.adversarial = adversarial
        
        # Labels
        
        with open("testset_stats", "rb") as fp:
            target_stats = pickle.load(fp)
        target_stats = np.array(target_stats)
        
        gt_label = target_stats[:self.length]
        
        self.alphas = alphas.split(",")
        
        train_gt_label = list(gt_label[:int(self.length * self.split_ratio)]) * len(alphas)
        test_gt_label = list(gt_label[int(self.length * self.split_ratio):]) * len(alphas)
        
        
        if(self.split == 'train'):
            self.gt_label = train_gt_label
        else:
            self.gt_label = test_gt_label
        
        # Image Locations
        
        train_image_locs = []
        test_image_locs = []
        for alpha in self.alphas:
            image_locs = get_image_loc(alpha=alpha, length=length, adversarial=False, data_dir=data_dir)
            train_locs = image_locs[:int(self.length * self.split_ratio)]
            test_locs = image_locs[int(self.length * self.split_ratio):]
            train_image_locs.extend(train_locs)
            test_image_locs.extend(test_locs)
        
            
        adversarial_train_image_locs = []
        adversarial_test_image_locs = []
        for alpha in self.alphas:
            image_locs = get_image_loc(alpha=alpha, length=length, adversarial=True)
            train_locs = image_locs[:int(self.length * self.split_ratio)]
            test_locs = image_locs[int(self.length * self.split_ratio):]
            adversarial_train_image_locs.extend(train_locs)
            adversarial_test_image_locs.extend(test_locs)
            
        
        if(self.split == 'train'):
            self.image_locs = train_image_locs
            self.adversarial_image_locs = adversarial_train_image_locs
        else:
            self.image_locs = test_image_locs
            self.adversarial_image_locs = adversarial_test_image_locs
        
    def __len__(self):
        return len(self.image_locs)
        
    def __getitem__(self, index):
        
        if(self.split == 'train'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
                
        if(self.split == 'train'):
            if(random.random() < 0.8):
                img_loc = self.image_locs[index]
            else:
                img_loc = self.adversarial_image_locs[index]
        else:
            if(self.adversarial):
                img_loc = self.adversarial_image_locs[index]
            else:
                img_loc = self.image_locs[index]
            

        img = Image.open(img_loc)
        
        image = transform(img)
        label = self.gt_label[index]
        
        return image, label

trainset = AdversarialDataset(length=200)
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=0)

testset = AdversarialDataset(length=200, split='test')
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=0)


trainset = AdversarialDataset(split='train', data_dir='adversarial_images_2', length=200, split_ratio=0.8, adversarial=True, alphas="0.0,1.0")
train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=0)

testset = AdversarialDataset(split='test', data_dir='adversarial_images_2', length=200, split_ratio=0.8, adversarial=True, alphas="0.0,0.25,0.5,0.75,1.0")
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=0)

    
        
        
        
        
        
        
        
    
    