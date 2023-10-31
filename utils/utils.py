#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import torch
import os
import sys
import shutil
import numpy as np
import logging
import json
from omegaconf import OmegaConf, open_dict

from bunch import Bunch

from sklearn.model_selection import train_test_split


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(yaml_file):
    """
    解析yaml文件
    :param yaml_file: 配置文件
    :return: 配置类
    """
    config = yaml_file
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.log_dir = os.path.join("experiments", config.exp_name, "logs/")  # 日志
        config.cp_dir = os.path.join("experiments", config.exp_name, "checkpoints/")  # 模型
        config.img_dir = os.path.join("experiments", config.exp_name, "images/")  # 网络
        config.dpy_dir = os.path.join("experiments", config.exp_name, "deploy/")  # 部署

    mkdir_if_not_exist(config.log_dir)  # 创建文件夹
    mkdir_if_not_exist(config.cp_dir)  # 创建文件夹
    mkdir_if_not_exist(config.img_dir)  # 创建文件夹
    mkdir_if_not_exist(config.dpy_dir)  # 创建文件夹
    return config


def prepare_device(config):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu_use = config.n_gpu_use
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_logger(filename, verbosity=1, name=None):
    # logging.basicConfig(level=logging.DEBUG, stream=None)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    sh.setLevel(logging.INFO)

    logger.addHandler(sh)

    return logger


def split_dataset(data_lst, label_lst, train_size=0.8, random_state=100):
    data_train, data_val_test = train_test_split(np.array(data_lst),
                                                 train_size=train_size,
                                                 random_state=random_state)
    label_train, label_val_test = train_test_split(np.array(label_lst),
                                                   train_size=train_size,
                                                   random_state=random_state)
    data_val, data_test = train_test_split(data_val_test,
                                           train_size=0.5,
                                           random_state=random_state)
    label_val, label_test = train_test_split(label_val_test,
                                             train_size=0.5,
                                             random_state=random_state)

    return data_train, label_train, data_val, label_val, data_test, label_test
