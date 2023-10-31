#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/10/9 9:18
"""
import torch
import torch.nn as nn

def lstm_forward(inp, initial_states, w_ih, w_hh, b_ih, b_hh):
    """
    input [bs, T, input_size]
    """
    h0, c0 = initial_states
    bs, seq_len, i_size = inp.shape
    h_size = w_ih.shape[0] // 4 # w_ih [4 * h_dim, i_size]
    bw_ih = w_ih.unsqueeze(0).tile(bs, 1, 1) # [bs, 4 * h_dim, i_size]
    bw_hh = w_hh.unsqueeze(0).tile(bs, 1, 1) # [bs, 4 * h_dim, h_dim]
    prev_h = h0 # [bs, h_d]
    prev_c = c0
    output_size = h_size
    output = torch.randn(bs, seq_len, output_size)

    # 对时间进行遍历
    for t in range(seq_len):
        x = inp[:, t, :] # [bs, input_size]
        # 为了能进行bmm，对x增加一维 [bs, i_s, 1]
        w_times_x = torch.bmm(bw_ih, x.unsqueeze(-1)).squeeze(-1) # [bs, 4h_d]
        w_times_h_prev = torch.bmm(bw_hh, prev_h.unsqueeze(-1)).squeeze(-1) # [bs, 4h_d]
        # 计算i门，取矩阵的前1/4
        i = 0
        i_t = torch.sigmoid(w_times_x[:, h_size*i:h_size*(1+i)] + w_times_h_prev[:,h_size*i:h_size*(1+i)] + b_ih[h_size*i:h_size*(1+i)] + b_hh[h_size*i:h_size*(1+i)])
        # f门
        i += 1
        f_t = torch.sigmoid(w_times_x[:, h_size*i:h_size*(1+i)] + w_times_h_prev[:, h_size*i:h_size*(1+i)] + b_ih[h_size*i:h_size*(1+i)] + b_hh[h_size*i:h_size*(1+i)])
        # g门
        i += 1
        g_t = torch.tanh(w_times_x[:, h_size*i:h_size*(1+i)] + w_times_h_prev[:, h_size*i:h_size*(1+i)] + b_ih[h_size*i:h_size*(1+i)] + b_hh[h_size*i:h_size*(1+i)])
        # o门
        i += 1
        o_t = torch.sigmoid(w_times_x[:, h_size*i:h_size*(1+i)] + w_times_h_prev[:, h_size*i:h_size*(1+i)] + b_ih[h_size*i:h_size*(1+i)] + b_hh[h_size*i:h_size*(1+i)])

        # cell
        prev_c = f_t * prev_c + i_t * g_t

        # h
        prev_h = o_t * torch.tanh(prev_c)

        output[:, t, :] = prev_h

    return output, (prev_h, prev_c)


def test_lstm_impl():
    bs, t, i_size, h_size = 2, 3, 4, 5
    inp = torch.randn(bs, t, i_size)
    # 不需要训练
    c0 = torch.randn(bs, h_size)
    h0 = torch.randn(bs, h_size)

    # 调用官方API
    lstm_layer = nn.LSTM(i_size, h_size, batch_first=True)
    output, _ = lstm_layer(inp, (h0.unsqueeze(0), c0.unsqueeze(0)))
    for k, v in lstm_layer.named_parameters():
        print(k, "# #", v.shape)

    print("++++++++++++++++++++++++++++++++++++++")

    w_ih = lstm_layer.weight_ih_l0
    w_hh = lstm_layer.weight_hh_l0
    b_ih = lstm_layer.bias_ih_l0
    b_hh = lstm_layer.bias_hh_l0

    output2, _ = lstm_forward(inp, (h0, c0), w_ih, w_hh, b_ih, b_hh)
    print(torch.allclose(output2, output))
    print(output)
    print(output2)

test_lstm_impl()
