#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/28 9:51
"""

from pyexpat import model
import os
import argparse
import time
import onnxruntime
from omegaconf import OmegaConf
from models.ADSB_model import ADSBModel

import torch
import numpy as np
from utils.utils import process_config, prepare_device, get_logger
from infers.ADSB_infer import Infer


def pytorch_eval(args, config, logger, dummy_data):

    logger.info('构造网络...')
    device, device_ids = prepare_device(config=config)
    model = ADSBModel(config=config).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    assert args.pretrained_model_path is not None, '请指定预训练模型路径！'
    logger.info('加载pth网络参数...')
    model_path = args.pretrained_model_path
    model.load_state_dict(torch.load(model_path))

    dummy_data = torch.Tensor(dummy_data).to(device)
    logger.info('开始推理...')
    time1 = time.time()
    for i in range(args.loop):
        time_bs1 = time.time()
        with torch.no_grad():
            out_img = model(dummy_data)
            out_img_numpy = out_img.detach().cpu().numpy()
        time_bs2 = time.time()
        time_use_pt_bs = time_bs2 - time_bs1
        if i % 10 == 0:
            logger.info(f'[%d/%d]: PyTorch time spent {time_use_pt_bs}', i, args.loop)
    time2 = time.time()
    time_use_pt = time2 - time1
    logger.info(f'PyTorch total time spent {time_use_pt} for {args.loop} loops,'
                f' and FPS={args.loop * dummy_data.shape[0] // time_use_pt}')

def onnx_eval(args, config, logger, dummy_data):
    logger.info(onnxruntime.get_device())
    logger.info('加载onnx网络参数...')
    sess = onnxruntime.InferenceSession(args.onnx_path,
                                        providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    logger.info('开始推理...')
    time1 = time.time()
    for i in range(args.loop):
        time_bs1 = time.time()
        out_ort_img = sess.run(None, {sess.get_inputs()[0].name: dummy_data, })
        time_bs2 = time.time()
        time_use_onnx_bs = time_bs2 - time_bs1
        if i % 10 == 0:
            logger.info(f'[%d/%d]: ONNX time spent {time_use_onnx_bs}', i, args.loop)
    time2 = time.time()
    time_use_onnx = time2 - time1
    logger.info(f'ONNX total time spent {time_use_onnx} for {args.loop} loops,'
                f' and FPS={args.loop * dummy_data.shape[0] // time_use_onnx}')


def onnx_eval_main(args):
    config = OmegaConf.load(args.config)

    logger = get_logger(os.path.join(r'../experiments/ADS-B/logs', 'onnx_eval_log.log'))
    logger.info('解析配置...')

    logger.info('加载数据...')
    dummy_data = np.ones((1, 2, 4800), dtype=np.float32)

    logger.info('Pytorch 推理...')
    pytorch_eval(args, config, logger, dummy_data)
    logger.info('Pytorch 推理完成...')

    logger.info('ONNX 推理...')
    onnx_eval(args, config, logger, dummy_data)
    logger.info('ONNX 推理完成...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str,
                        default=r"../experiments/ADS-B/checkpoints/ADS-B_epochs_30_acc_90.5.pth")
    parser.add_argument("--config", type=str, default=r"../configs/config.yaml")
    parser.add_argument("--onnx_path", type=str, default='../experiments/ADS-B/deploy/net_bs30_v1.onnx')
    parser.add_argument("--loop", type=int, default=100)

    args = parser.parse_args()
    onnx_eval_main(args)
