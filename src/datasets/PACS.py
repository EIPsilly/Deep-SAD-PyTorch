import argparse
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import DataLoader
import glob
import torch
import logging
import os
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from collections import Counter
import yaml

with open("/home/hzw/DGAD/domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

domain_to_idx = config["PACS_domain_to_idx"]
class_to_idx = config["PACS_class_to_idx"]
idx_to_class = config["PACS_idx_to_class"]

# 计算不同domain 和 class 对应的labal
def calc_label_idx(domain, idx):
    return domain_to_idx[domain] * 7 + idx


class PACSDataset(torch.utils.data.Dataset):
    def __init__(self, root, normal_class, anomaly_class, train = True):
        data = np.load(root)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 253

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        if train == True:
            # load dataset
            self.img_paths = data["train_set_path"]
            self.semi_targets = data["train_labels"]
            self.semi_targets[np.where(self.semi_targets != -1)[0]] %= 7
            self.labels = np.array([class_to_idx[path.split("/")[-2]] for path in self.img_paths])
            
            # self.semi_targets_transform = transforms.Lambda(lambda x: -1 if (x in anomaly_class) else (1 if (x in normal_class) else 0))
            self.semi_targets = np.array([-1 if x in anomaly_class else (1 if x in normal_class else 0) for x in self.semi_targets])
            # self.labels_transform = transforms.Lambda(lambda x: int(x in anomaly_class))
            self.labels = np.array([int(x in anomaly_class) for x in self.labels])
        else:
            self.img_paths = data["test_set_path"]
            self.labels = np.array([class_to_idx[path.split("/")[-2]] for path in self.img_paths])
            self.labels = np.array([int(x in anomaly_class) for x in self.labels])

            self.semi_targets = self.labels
        print(Counter(self.labels))
        print(Counter(self.semi_targets))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path= self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        semi_target = self.semi_targets[idx]
        
        return img, label, semi_target, idx

class PACS_Data():
    def __init__(self, args):
        if isinstance(args, argparse.Namespace):
            normal_class = args.normal_class
            anomaly_class = args.anomaly_class
        else:
            normal_class = args["normal_class"]
            anomaly_class = args["anomaly_class"]
        
        root = f'/home/hzw/DGAD/domain-generalization-for-anomaly-detection/data/20231204-PACS-{"".join(list(map(str,normal_class)))}-{"".join(list(map(str,anomaly_class)))}.npz'
        self.train_set = PACSDataset(root, normal_class, anomaly_class, train = True)
        self.test_set = PACSDataset(root, normal_class, anomaly_class, train = False)
    
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 8, drop_last_train = True):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=drop_last_train)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader

if __name__ == "__main__":
    
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--normal_class", nargs="+", type=int, default=[0,1,2,3])
    args.add_argument("--anomaly_class", nargs="+", type=int, default= [4,5,6])
 
    args = args.parse_args()
    
    train_loader, test_loader = PACS_Data(args).loaders(16)

    label_list = []
    semi_target_list = []
    for batch in train_loader:
        img, label, semi_target, idx = batch
        label_list.append(label)
        semi_target_list.append(semi_target)
    
    print("label_list\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    print("semi_target_list\t" + str(dict(sorted(Counter(np.concatenate(semi_target_list)).items()))))

    label_list = []
    semi_target_list = []
    for batch in test_loader:
        img, label, semi_target, idx = batch
        label_list.append(label)
        semi_target_list.append(semi_target)
    
    print("label_list\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    print("semi_target_list\t" + str(dict(sorted(Counter(np.concatenate(semi_target_list)).items()))))