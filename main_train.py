#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import os
import torch
import wandb
import argparse

from omegaconf import OmegaConf

from data_loaders.my_ADSB_dl import MyADSBDataLoader
from models.ADSB_model import ADSBModel
from trainers.ADSB_trainer import Trainer

from utils.utils import process_config, prepare_device, get_logger


os.environ["WANDB_API_KEY"] = '2ec38f4f7afe80b121961f27c7b085e640e73165'


def main_train(args):
    config = OmegaConf.load(args.config)
    config = process_config(config)

    logger = get_logger(os.path.join(config.log_dir, 'train_log.log'))
    logger.info('解析配置...')

    wandb.init(
        project="ADS-B-Detection",
        config={
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "lr": config.optimizer.lr,
            "decay_rate": config.optimizer.decay
        }
    )

    logger.info('加载数据...')
    dl = MyADSBDataLoader(config=config, logger=logger)

    logger.info('构造网络...')
    device, device_ids = prepare_device(config=config)
    model = ADSBModel(config=config).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info('训练网络...')
    trainer = Trainer(
        model      = model,
        train_data = dl.get_train_data(),
        val_data   = dl.get_val_data(),
        config     = config,
        logger     = logger,
        device     = device)
    # wandb.watch(model, log="all")
    trainer.train()
    wandb.finish()
    logger.info('训练完成...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--config",                type=str, default="configs/config.yaml")

    args = parser.parse_args()
    main_train(args)
