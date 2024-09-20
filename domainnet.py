from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data

class DomainNet(data.Dataset):
    def __init__(self, root, imbalanced, domain, train=True, transform=None, from_file=False):
        
        if not from_file:
            data = []
            labels = []
            
            if imbalanced == "mini":
                
                if train:
                    split = "train"
                else:
                    split = "test"
                
                file_path = os.path.join('..','SSISFDA','data','Imbalanced','domainNet_mini','%s_%s_mini.txt' % (domain, split))
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
            
            elif imbalanced == "126":
                file_path = os.path.join('..','SSISFDA','data','Imbalanced','domainNet_126','%s_126.txt' % domain)
                with open(file_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        path, label = line.split(" ")
                        sample = os.path.join(root, path)
                        data.append(sample)
                        labels.append(int(label))

                test_perc = 20
                test_len = len(self.data)*test_perc//100                   

                if train:
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
        return len(self.X)

