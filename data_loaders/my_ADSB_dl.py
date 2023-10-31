#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import os
import glob
import torch
import numpy as np

from utils.utils import process_config, split_dataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from scipy.io import loadmat


def get_data_form_path(path):
    signal_lst = []
    label_lst = []
    for class_path in path:
        data_path_lst = glob.glob(os.path.join(class_path, '*'))
        for data_path in data_path_lst:
            data = loadmat(data_path)
            signal = data['data'].T
            labels = int(data['Label'][0])
            signal_lst.append(signal.astype(np.float32))
            label_lst.append(labels)
    return signal_lst, label_lst


class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.data)


class MyADSBDataLoader(Dataset):
    def __init__(self, config, logger):
        super(MyADSBDataLoader, self).__init__()
        self.config = config
        self.logger = logger
        self.dl_config = config.dataloader

        class_path_lst = glob.glob(os.path.join(self.dl_config.data_path, '*'))

        signal_lst, label_lst = get_data_form_path(class_path_lst)

        self.Signal_train, self.Label_train, self.Signal_val, self.Label_val, self.Signal_test, self.Label_test = split_dataset(
            data_lst=np.array(signal_lst), label_lst=np.array(label_lst), train_size=self.dl_config.train_size,
            random_state=self.dl_config.random_state)

        self.logger.info(
            "Data_train: %s, Label_train: %s" % (str(self.Signal_train.shape), str(self.Label_train.shape)))
        self.logger.info("Data_val: %s, Label_val: %s" % (str(self.Signal_val.shape), str(self.Label_val.shape)))
        self.logger.info("Data_test: %s, Label_test: %s" % (str(self.Signal_test.shape), str(self.Label_test.shape)))

    def get_train_data(self):
        train_dataset = MyDataset(self.Signal_train, self.Label_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size,
                                                   shuffle=True, num_workers=16)
        train_num = len(train_dataset)
        return train_loader

    def get_val_data(self):
        val_dataset = MyDataset(self.Signal_val, self.Label_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size,
                                                 shuffle=False, num_workers=16)
        val_num = len(val_dataset)
        return val_loader, val_num

    def get_test_data(self):
        test_dataset = MyDataset(self.Signal_test, self.Label_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
        test_num = len(test_dataset)
        return test_loader


if __name__ == "__main__":
    config = process_config(r'/home/zxiat/Jet/Project/DL-Project-Template-master/configs/config.json')
    dl = MyADSBDataLoader(config=config)
    train_loader, train_num = dl.get_train_data()
    print(train_loader, train_num)
