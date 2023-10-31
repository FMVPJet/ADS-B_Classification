#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import torch
import torch.nn as nn

from models.base_model.resnet1d import ResNet1D

class ADSBModel(nn.Module):

    def __init__(self, config):
        super(ADSBModel, self).__init__()
        self.model_config = config.model
        self.model        = None
        self.build_model()

    def build_model(self):
        self.model = ResNet1D(in_channels        = self.model_config.in_channels,
                              base_filters       = self.model_config.base_filters,
                              kernel_size        = self.model_config.kernel_size,
                              stride             = self.model_config.stride,
                              n_block            = self.model_config.n_block,
                              groups             = self.model_config.groups,
                              n_classes          = self.model_config.n_classes,
                              downsample_gap     = self.model_config.downsample_gap,
                              increasefilter_gap = self.model_config.increasefilter_gap,
                              verbose            = False)

    def forward(self, x):
        x = self.model(x)
        return x
