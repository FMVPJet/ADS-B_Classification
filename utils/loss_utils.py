#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""

import torch.nn as nn


def loss_func():
    loss_function = nn.CrossEntropyLoss()
    return loss_function
