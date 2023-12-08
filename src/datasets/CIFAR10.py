import glob
import os
import argparse
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
import logging
import yaml

with open("/home/hzw/DGAD/domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)


class CIFAR10_Dataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None, augment_transform = None):
        self.data = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        augimg = img

        if self.augment_transform is not None:
            augimg = self.augment_transform(Image.fromarray(augimg, mode="RGB"))

        if self.transform is not None:
            img = self.transform(Image.fromarray(img, mode="RGB"))

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, label, idx

class CIFAR10_Data():

    def __init__(self, args):

        # normal_class = args["normal_class"]
        # anomaly_class = args["anomaly_class"]
        # contamination_rate = args["contamination_rate"]
        # labeled_rate = args["labeled_rate"]
        # seed = args["seed"]
        # train_binary = args["train_binary"]
        # corrupt_type = args["corrupt_type"]

        normal_class = args.normal_class
        anomaly_class = args.anomaly_class
        contamination_rate = args.contamination_rate
        labeled_rate = args.labeled_rate
        seed = args.seed
        train_binary = args.train_binary
        corrupt_type = args.corrupt_type

        self.normal_class = normal_class
        self.anomaly_class = anomaly_class
        self.contamination_rate = contamination_rate
        self.labeled_rate = labeled_rate
        self.seed = seed

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = 3
        crop_pct = 0.8
        image_size = 32

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        augment_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_target_transform = transforms.Lambda(lambda x: int(x in anomaly_class))
        train_target_transform = transforms.Lambda(lambda x: -1 if (x in anomaly_class) else (1 if (x in normal_class) else 0))

        train_set = CIFAR10(root = config["cifar_10_root"], train=True, download=True)
        train_set.targets = np.array(train_set.targets)

        normal_idx = np.where(np.isin(train_set.targets, normal_class))[0]
        sample_cnt = int(len(normal_idx) / (1 - (labeled_rate / (len(normal_class) + len(anomaly_class)) * len(anomaly_class) + contamination_rate)))
        labeled_cnt = int(sample_cnt * labeled_rate)
        contamination_cnt = int(sample_cnt * contamination_rate)
        
        # 从normal中选出labeled数据
        labeled_idx = []
        for i in normal_class + anomaly_class:
            idx = np.where(train_set.targets == i)[0]
            np.random.seed(self.seed)
            idx = np.random.choice(idx, int(labeled_cnt / len(normal_class + anomaly_class)), False)
            labeled_idx.append(idx)
        labeled_idx = np.concatenate(labeled_idx)

        # 从anomaly中选出污染数据
        contamination_idx = []
        for i in anomaly_class:
            idx = np.where(train_set.targets == i)[0]
            # 除去idx中，作为labeled出现过的anomaly下标
            idx = idx[~np.isin(idx, labeled_idx)]
            np.random.seed(self.seed)
            idx = np.random.choice(idx, int(contamination_cnt / len(anomaly_class)), False)
            contamination_idx.append(idx)
        contamination_idx = np.concatenate(contamination_idx)

        # 无标签数据标记为 -1
        # normal中的无标签数据标记为 -1
        mask = normal_idx[~np.isin(normal_idx, labeled_idx)]
        train_set.targets[mask] = -1
        # anomaly中的无标签数据，即污染数据 标记为 -1
        train_set.targets[contamination_idx] = -1
        idx = np.union1d(np.union1d(normal_idx, labeled_idx), contamination_idx)

        x_train = train_set.data[idx]
        y_train = train_set.targets[idx]

        logging.info(Counter(y_train))

        # 从训练集中划分出验证集
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=self.seed, stratify=y_train)

        logging.info("y_train\t" + str(Counter(y_train)))
        logging.info("y_val\t" + str(Counter(y_val)))
        
        # 如果训练集标签只考虑 正常 和 异常 两种
        if train_binary:
            self.train_set = CIFAR10_Dataset(x_train, y_train, transform=train_transform, target_transform=train_target_transform, augment_transform = augment_transform)
        else: # 训练集标签考虑 多类别的情况
            self.train_set = CIFAR10_Dataset(x_train, y_train, transform=train_transform, augment_transform = augment_transform)
        
        self.val_set = CIFAR10_Dataset(x_val, y_val, transform=test_transform, target_transform=train_target_transform, augment_transform = augment_transform)

        # 测试集选择原来的数据集
        if corrupt_type == "origin":
            test_set = CIFAR10(root = config["cifar_10_root"], train=False, download=True)
            self.test_set = CIFAR10_Dataset(test_set.data, test_set.targets, transform=test_transform, target_transform=test_target_transform)
            logging.info("y_test\t" + str(Counter(test_set.targets)))
        else:   # 测试集使用ood的数据集
            ood_path = config["cifar_10_ood_root"] + f'/cifar10_5_{corrupt_type}/test'
            x_test = []
            y_test = []
            for _class_, label in train_set.class_to_idx.items():
                img_paths = glob.glob(os.path.join(ood_path, _class_) + "/*.png")
                for img_path in img_paths:
                    img = Image.open(img_path).convert("RGB")
                    x_test.append(np.array(img))
                    y_test.append(label)
                    
            x_test, y_test = np.array(x_test), np.array(y_test)
            self.test_set = CIFAR10_Dataset(x_test, y_test, transform=test_transform, target_transform=test_target_transform)
            logging.info("y_test\t" + str(Counter(y_test)))

        # x_test = test_set.data
        # y_test = test_set.targets
        # self.test_set = MyDataset(x_test, y_test, transform=test_transform, target_transform=test_target_transform)


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, drop_last_train = True):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=drop_last_train)

        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=drop_last_train)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
        # return train_loader, val_loader, test_loader

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="cifar10")
    args.add_argument("--normal_class", nargs="+", type=int, default=[0,1,2,3,4,5,6])
    args.add_argument("--anomaly_class", nargs="+", type=int, default= [7,8,9])
    args.add_argument("--contamination_rate", type=float ,default=0.02)
    args.add_argument("--labeled_rate", type=float, default=0.02)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_binary", type=bool, default=True)
    args.add_argument("--corrupt_type", type=str, default="brightness", choices=['brightness','contrast','defocus_blur','gaussian_noise', 'origin'])
 
    args = args.parse_args(["--data_name", "cifar10",
                            "--normal_class", "0","1","2","3","4","5","6",
                            "--anomaly_class", "7","8","9",
                            "--train_binary", "True",
                            "--corrupt_type", "brightness",
                            ])

    data = CIFAR10_Data(args)

    train_loader, test_loader = data.loaders(batch_size = 128)

    label_list = []
    semi_targets_list = []
    for batch in train_loader:
        inputs, labels, semi_targets, idx = batch
        label_list.append(labels)
        semi_targets_list.append(semi_targets)
    
    print("====train_loader====")
    print(Counter(np.concatenate(label_list)))
    print(Counter(np.concatenate(semi_targets_list)))

    # label_list = []
    # semi_targets_list = []
    # for batch in val_loader:
    #     inputs, labels, semi_targets, idx = batch
    #     label_list.append(labels)
    #     semi_targets_list.append(semi_targets)
    
    # print("====val_loader====")
    # print(Counter(np.concatenate(label_list)))
    # print(Counter(np.concatenate(semi_targets_list)))

    
    label_list = []
    semi_targets_list = []
    for batch in test_loader:
        inputs, labels, semi_targets, idx = batch
        label_list.append(labels)
        semi_targets_list.append(semi_targets)
    
    print("====test_loader====")
    print(Counter(np.concatenate(label_list)))
    print(Counter(np.concatenate(semi_targets_list)))