from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data

class OfficeHome(data.Dataset):
    def __init__(self, root, imbalanced, domain, train=True, transform=None, from_file=False):
        self.train = train
        self.transform = transform
        
        if not from_file:
            data = []
            labels = []
            
            if imbalanced:
                if imbalanced == "_RS" and self.train:
                    file_path = '../SSISFDA/data/Imbalanced/officeHome_RSUT/%s%s.txt' % (domain, imbalanced)
                elif imbalanced == "_RS" and not self.train:
                    file_path = '../SSISFDA/data/Imbalanced/officeHome_RSUT/%s_UT.txt' % (domain)
                elif imbalanced == "_UT" and self.train:
                    file_path = '../SSISFDA/data/Imbalanced/officeHome_RSUT/%s%s.txt' % (domain, imbalanced)
                elif imbalanced == "_UT" and not self.train:
                    file_path = '../SSISFDA/data/Imbalanced/officeHome_RSUT/%s_RS.txt' % (domain)
                else:
                    print("ERROR: Unknown configuration %s %s" % (domain, imbalanced))
            else:
                file_path = '../SSISFDA/data/Imbalanced/officeHome/%s.txt' % (domain)
            #print(file_path);exit()
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

