#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/9/19 17:22
"""
import torch
import time
import warnings

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Infer(object):
    def __init__(self, model, test_data, config, logger, device):
        self.model       = model
        self.test_data   = test_data
        self.config      = config
        self.device      = device
        self.label       = []
        self.infer_label = []
        self.logger      = logger

    def infer(self):
        start_time = time.time()
        self.model.eval()
        for test_step, (test_data, test_label) in enumerate(self.test_data):
            test_data   = test_data.to(self.device)
            test_label  = int(test_label)

            output      = self.model(test_data)
            output      = torch.softmax(output, dim=1)
            infer_label = int(torch.argmax(output, dim=1).cpu())

            self.label.append(test_label)
            self.infer_label.append(infer_label)
            self.logger.info('No.%s: 真实Label: %s, 预测Label: %s' % (test_step, test_label, infer_label))

        warnings.filterwarnings("ignore")
        accuracy = accuracy_score(self.label, self.infer_label) * 100
        precision, recall, f_score, _ = precision_recall_fscore_support(
            self.label, self.infer_label, average='macro')
        self.logger.info("预测结果: Accuracy: %0.4f%%" % accuracy)
        self.logger.info("预测结果: F1 score: %0.4f, Precision: %0.4f, recall %0.4f" % (f_score, precision, recall))

        end_time = time.time()
        self.logger.info("预测用时: %0.4fs" % (end_time - start_time))
