#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import os
import argparse
import torch

from omegaconf import OmegaConf

from data_loaders.my_ADSB_dl import MyADSBDataLoader
from infers.ADSB_infer import Infer
from models.ADSB_model import ADSBModel

from utils.utils import process_config,prepare_device, get_logger



def main_test(args):
    config = OmegaConf.load(args.config)
    config = process_config(config)

    logger = get_logger(os.path.join(config.log_dir, 'infer_log.log'))
    logger.info('解析配置...')

    logger.info('加载模型...')
    dl = MyADSBDataLoader(config=config, logger=logger)

    logger.info('构造网络...')
    device, device_ids = prepare_device(config=config)
    model = ADSBModel(config=config).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    assert args.pretrained_model_path is not None, '请指定预训练模型路径！'
    logger.info('加载网络参数...')
    model_path = args.pretrained_model_path
    model.load_state_dict(torch.load(model_path))

    logger.info('预测数据...')
    inference = Infer(
        model     = model,
        test_data = dl.get_test_data(),
        config    = config,
        logger    = logger,
        device    = device)
    inference.infer()
    logger.info('预测完成...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default=r"experiments/ADS-B/checkpoints/ADS-B_epochs_0_acc_58.5.pth")
    parser.add_argument("--config",                type=str, default=r"configs/config.yaml")

    args = parser.parse_args()
    main_test(args)
