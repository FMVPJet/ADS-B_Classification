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
import time
import wandb

from utils.loss_utils import loss_func


class Trainer(object):
    def __init__(self, model, train_data, val_data, config, logger, device):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.logger = logger
        self.device = device

        self.optimizer()
        self.logger.info(self.optimizer().__class__.__name__)

        self.criterion = loss_func()

    def optimizer(self):
        optimizer_config = self.config.optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=optimizer_config.lr,
                                     weight_decay=optimizer_config.decay)
        return optimizer


    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            start_time = time.time()
            for train_step, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device).to(torch.int64)

                self.optimizer().zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer().step()

                if train_step % self.config.log_interval == 0:
                    self.logger.info('Epoch: {}/{}, train_step: {}/{}, loss: {:.4f}'.format(
                        epoch,
                        self.config.num_epochs,
                        train_step + self.config.log_interval,
                        len(self.train_data),
                        loss.item()))
                wandb.log({"loss": loss})
                # rate = (train_step + 1) / len(self.train_data)
                # a = "*" * int(rate * 50)
                # b = "." * int((1 - rate) * 50)
                # print("[INFO] train loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="\n")
            if self.val_data[0] is not None:
                self.model.eval()
                acc = 0.0
                for eval_step, (data, target) in enumerate(self.val_data[0]):
                    data = data.to(self.device)

                    with torch.no_grad():
                        output = self.model(data)
                    output = torch.softmax(output, dim=1)
                    predict = torch.max(output, dim=1)[1].cpu()

                    acc += (predict == target).sum().item()
                val_accuracy = (acc / self.val_data[1]) * 100
                save_path = os.path.join(self.config.cp_dir,
                                         self.config.exp_name + '_epochs_' + str(
                                             epoch) + '_acc_' + str(val_accuracy)[:5] + '.pth')
                # print('[INFO] Accuracy: %.3f%%' % val_accuracy)
                self.logger.info('Accuracy: %.3f%%' % val_accuracy)
                wandb.log({"acc": val_accuracy})
                torch.save(self.model.state_dict(), save_path)
            end_time = time.time()
            # print("[INFO] 训练用时: %.3f s" % (end_time - start_time), '\n')
            self.logger.info("训练用时: %.3f s" % (end_time - start_time))
            print('==' * 30)
