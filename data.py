"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from PIL import Image
import glob
import sklearn.metrics
from torch.utils.data import Dataset
import pickle
import numpy as np
import random


crop_size = 32
padding = 4

def get_image_loc(alpha='1.0', adversarial=True, data_dir="adversarial_images", length=1200):

    last_str_match = "_orig_0.png"
    if adversarial:
        last_str_match = "_ILFO_*.png"

    all_images = []
    
    print(data_dir)
    
    for i in tqdm(range(length)):
        for j in range(2):
            if(adversarial):
                img_loc = glob.glob(data_dir+"/output_"+alpha+"/img_"+str(i)+"_"+str(j)+last_str_match)[0]
            else:
                img_loc = glob.glob(data_dir+"/orig/img_"+str(i)+"_"+str(j)+last_str_match)[0]
                
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
        
        train_gt_label = list(gt_label[:int(self.length * self.split_ratio)]) * len(self.alphas)
        test_gt_label = list(gt_label[int(self.length * self.split_ratio):]) * len(self.alphas)
        
        
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
            image_locs = get_image_loc(alpha=alpha, length=length, adversarial=True, data_dir=data_dir)
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
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip(),
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
            if(random.random() > 1/len(self.alphas)):
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
    

class FullDataset(Dataset):
    
    def __init__(self, split='train', data_dir='adversarial_images', length=1200, split_ratio=0.8, adversarial=True, alphas="0.0,0.25,0.5,0.75,1.0"):
        
        self.split = split
        self.length = length*2
        self.split_ratio = split_ratio
        self.adversarial = adversarial
        
        # Labels
        
        
        with open("trainset_stats", "rb") as fp:
            train_stats = pickle.load(fp)
        train_stats = np.array(train_stats)
        
        with open("testset_stats", "rb") as fp:
            target_stats = pickle.load(fp)
        target_stats = np.array(target_stats)
        
        self.alphas = alphas.split(",")
        
        train_gt_label = train_stats
        self.gt_label = train_gt_label
        
        self.test_labels = target_stats[:self.length]
        
        # Image Locations
        
        train_image_locs = ['train_images/' + str(i) + '.png' for i in range(1, 50001)]
        
        adversarial_train_image_locs = []
        adversarial_test_image_locs = []
        for alpha in self.alphas:
            image_locs = get_image_loc(alpha=alpha, length=length, adversarial=True, data_dir=data_dir)
            train_locs = image_locs[:int(self.length * self.split_ratio)]
            test_locs = image_locs[int(self.length * self.split_ratio):]
            adversarial_train_image_locs.extend(train_locs)
            adversarial_test_image_locs.extend(test_locs)
            
        self.adversarial_len = len(adversarial_train_image_locs)
        
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
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip(),
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
            if(random.random() > 0.1):
                img_loc = self.image_locs[index]        
                label = self.gt_label[index]
            else:
                idx = random.randint(0, self.adversarial_len-1)
                img_loc = self.adversarial_image_locs[idx]
                label = self.test_labels[idx]
        else:
            if(self.adversarial):
                img_loc = self.adversarial_image_locs[index]
            else:
                img_loc = self.image_locs[index]
            

        img = Image.open(img_loc)
        
        image = transform(img)
        
        return image, label
    



def prepare_train_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4, size=1200,
                       alphas="0.0,0.25,0.5,0.75,1.0", acc_adv=False, 
                       image_folder='images', full_train=False):
    
    if 'adversarial' in dataset:
        '''
        if(acc_adv):
            data_dir = 'adversarial_images_2'
        else:
            data_dir = 'adversarial_images'
        '''
        data_dir = image_folder
            
        if(full_train):
            trainset = FullDataset(length=size, alphas=alphas, data_dir=data_dir)
        else:
            trainset = AdversarialDataset(length=size, alphas=alphas, data_dir=data_dir)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
        

    elif 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4, adversarial=True, size=1200,
                      alphas="0.0,0.25,0.5,0.75,1.0", acc_adv=False, image_folder='images'):

    if 'adversarial' in dataset:
        '''
        if(acc_adv):
            data_dir = 'adversarial_images_2'
        else:
            data_dir = 'adversarial_images'
        '''
        data_dir = image_folder
        
        if(adversarial):
            testset = AdversarialDataset(length=size, split='test', alphas=alphas, data_dir=data_dir)
        else:
            testset = AdversarialDataset(length=size, split='test', adversarial=False, alphas=alphas, data_dir=data_dir)
        test_loader = torch.utils.data.DataLoader(testset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)

    
    elif 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4524,  0.4525,  0.4690),
                                         (0.2194,  0.2266,  0.2285)),
                ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/tmp/data',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    else:
        test_loader = None
    return test_loader
