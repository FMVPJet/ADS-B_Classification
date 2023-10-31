#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/28 8:54
"""
from pyexpat import model
import os
import argparse
from omegaconf import OmegaConf
from models.ADSB_model import ADSBModel

import torch
import numpy as np
from torchvision.models.resnet import resnet50
from utils.utils import process_config, prepare_device, get_logger

TORCH_WEIGHT_PATH = r"H:\pytorch-onnx-tensorrt\new_test\resnet50-0676ba61.pth"
ONNX_MODEL_PATH = r"H:\pytorch-onnx-tensorrt\new_test\net_bs8_v1.onnx"


def torch2onnx_main(args):

    config = OmegaConf.load(args.config)
    # config = process_config(config)

    logger = get_logger(os.path.join(r'../experiments/ADS-B/logs', 'torch2onnx_log.log'))
    logger.info('解析配置...')

    logger.info('加载数据...')
    data_input = np.ones((1, 2, 4800), dtype=np.float32)

    logger.info('构造网络...')
    device, device_ids = prepare_device(config=config)
    model = ADSBModel(config=config).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    assert args.pretrained_model_path is not None, '请指定预训练模型路径！'
    logger.info('加载网络参数...')
    model_path = args.pretrained_model_path
    model.load_state_dict(torch.load(model_path))

    logger.info('开始转换...')
    model.eval()
    dummy_data = torch.Tensor(data_input).to(device)
    torch.onnx.export(model=model,
                      args=dummy_data,
                      f=os.path.join(r'../experiments/ADS-B/deploy', args.onnx_name),
                      input_names=['input'],
                      output_names=['output'],
                      export_params=True,
                      verbose=True,
                      opset_version=12)

    logger.info('onnx模型转换完成！')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str,
                        default=r"../experiments/ADS-B/checkpoints/ADS-B_epochs_30_acc_90.5.pth")
    parser.add_argument("--config", type=str, default=r"../configs/config.yaml")
    parser.add_argument("--onnx_name", type=str, default='net_bs30_v1.onnx')


    args = parser.parse_args()
    torch2onnx_main(args)
