#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/28 9:48
"""

import onnx
from onnxsim import simplify

ONNX_MODEL_PATH = '../experiments/ADS-B/deploy/net_bs30_v1.onnx'
ONNX_SIM_MODEL_PATH = '../experiments/ADS-B/deploy/net_bs30_v1_simple.onnx'

if __name__ == "__main__":
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx_sim_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim_model, ONNX_SIM_MODEL_PATH)
    print('ONNX file simplified!')
