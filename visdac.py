from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch.utils.data as data
import glob

class VISDAC(data.Dataset):
    def __init__(self, root, imbalanced, domain, train=True, transform=None, from_file=False):
        self.train = train
        self.transform = transform
        
        if domain == 'source':
            domain = 'train'
        elif domain == 'target':
            domain = 'validation'
        else:
            print("Unknown domain: {}".format(domain))
        
        if not from_file:
            data = []
            labels = []
            
            if imbalanced:
                if domain == 'train':# and self.train:
                    file_path = '../SSISFDA/data/Imbalanced/VISDA-'+imbalanced+'_RSUT/train_RS.txt'
                #elif domain == 'train' and not self.train:
                #    file_path = '../SSISFDA/data/Imbalanced/VISDA-'+imbalanced+'_RSUT/train_UT.txt'
                elif domain == 'validation':
                    file_path = '../SSISFDA/data/Imbalanced/VISDA-'+imbalanced+'_RSUT/validation_UT.txt'
                else:
                    print("Unknown domain: {}".format(domain))
                with open(file_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        path, label = line.split(" ")
                        sample = os.path.join(root, "/".join(path.split("/")[-3:]))
                        data.append(sample)
                        labels.append(int(label))
            else:
                #NEWclass_names = ['aeroplane','bicycle','bus','car','horse','knife','motorcycle','person','plant','skateboard','train','truck']
                #NEWfor c in range(12):
                #NEW    files = glob.glob(os.path.join(root,domain,class_names[c],"*"))
                #NEW    data.extend(files)
                #NEW    labels.extend([c]*len(files))
                if domain == 'train':# and self.train:
                    file_path = '../SSISFDA/data/Imbalanced/VISDA-C/image_list_train.txt'
                elif domain == 'validation':
                    file_path = '../SSISFDA/data/Imbalanced/VISDA-C/image_list_val.txt'
                else:
                    print("Unknown domain: {}".format(domain))
                with open(file_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        path, label = line.split(" ")
                        sample = os.path.join(root, "/".join(path.split("/")[-3:]))
                        data.append(sample)
                        labels.append(int(label))

            np.random.seed(1234)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]
            
            test_perc = 20
            test_len = len(self.data)*test_perc//100
            if self.train:
                self.data = self.data[test_len:]
                self.labels = self.labels[test_len:]
            else:
                self.data = self.data[:test_len]
                self.labels = self.labels[:test_len]
        
        else:

            self.data = np.load(os.path.join(root, domain+"_imgs.npy"))
            self.labels = np.load(os.path.join(root, domain+"_labels.npy"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]          

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
         
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)

