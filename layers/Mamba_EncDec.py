import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
class EncoderLayer(nn.Module):
    def __init__(self, attention1, attention1_r, attention2, attention2_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention1 = attention1
        self.attention1_r = attention1_r
        self.attention2 = attention2
        self.attention2_r = attention2_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, idx_emb_1, idx_pro_1, idx_emb_2, idx_pro_2, attn_mask=None, tau=None, delta=None):

        bs, _, _ = x.shape
        # long feature
        x_l1 = torch.zeros_like(x)
        x_l2 = torch.zeros_like(x)
        for i in range(bs):
            x_l1[i, :, :] = x[i, idx_emb_1[i].long(), :]
        new_x1 = self.attention1(x_l1) + self.attention1_r(x_l1.flip(dims=[1])).flip(dims=[1])
        for i in range(bs):
            x_l2[i] = new_x1[i, idx_pro_1[i].long(), :]

        

        # Short feature
        x_s1 = torch.zeros_like(x)
        x_s2 = torch.zeros_like(x)
        for i in range(bs):
            x_s1[i] = x[i, idx_emb_2[i].long(), :]
        new_x2 = self.attention1(x_s1) + self.attention1_r(x_s1.flip(dims=[1])).flip(dims=[1])
        for i in range(bs):
            x_s2[i] = new_x2[i, idx_pro_2[i].long(), :]

        attn = 1

        x = x + x_l2 + x_s2
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx_emb_1, idx_pro_1, idx_emb_2, idx_pro_2, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, idx_emb_1, idx_pro_1, idx_emb_2, idx_pro_2, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, idx_emb_1, idx_pro_1, idx_emb_2, idx_pro_2, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, idx_emb_1, idx_pro_1, idx_emb_2, idx_pro_2, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer1(nn.Module):
    def __init__(self, attention1, attention1_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer1, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention1 = attention1
        self.attention1_r = attention1_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, idx_emb_1, idx_pro_1, N, attn_mask=None, tau=None, delta=None):

        bs, _, _ = x.shape
        # long feature
        x_l1 = torch.zeros_like(x)
        x_l2 = torch.zeros_like(x)
        for i in range(bs):
            x_l1[i, :N, :] = x[i,:N,:][idx_emb_1[i].long(), :]
            x_l1[i, N:, :] = x[i,N:,:]
        attn1 = self.attention1(x_l1)
        attn2 = self.attention1_r(x_l1.flip(dims=[1])).flip(dims=[1])
        new_x1 = attn1 + attn2
        for i in range(bs):
            x_l2[i, :N, :] = new_x1[i,:N,:][idx_pro_1[i].long(), :]
            x_l2[i, N:, :] = new_x1[i,N:,:]

        x = x + x_l2
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn1, attn2


class Encoder1(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder1, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx_emb_1, idx_pro_1, N, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns1 = []
        attns2 = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn1, attn2 = attn_layer(x, idx_emb_1, idx_pro_1, N, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns1.append(attn1.detach().cpu().numpy())
                attns2.append(attn2.detach().cpu().numpy())
            x, attn1, attn2 = self.attn_layers[-1](x, idx_emb_1, idx_pro_1, N, tau=tau, delta=None)
            attns1.append(attn1.detach().cpu().numpy())
            attns2.append(attn2.detach().cpu().numpy())
        else:
            for attn_layer in self.attn_layers:
                x, attn1, attn2 = attn_layer(x, idx_emb_1, idx_pro_1, N, attn_mask=attn_mask, tau=tau, delta=delta)
                attns1.append(attn1.detach().cpu().numpy())
                attns2.append(attn2.detach().cpu().numpy())

        if self.norm is not None:
            x = self.norm(x)

        return x, attns1, attns2

class EncoderLayer2(nn.Module):
    def __init__(self, attention1, attention1_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer2, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention1 = attention1
        self.attention1_r = attention1_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, idx_emb_1, idx_pro_1, N, attn_mask=None, tau=None, delta=None):

        bs, _, _ = x.shape
        # long feature
        x_l1 = torch.zeros_like(x)
        x_l2 = torch.zeros_like(x)

        for i in range(bs):
            x_l1[i, :N, :] = x[i,:N,:][idx_emb_1[i].long(), :]
            x_l1[i, N:, :] = x[i,N:,:]
        new_x1 = self.attention1(x_l1) + self.attention1_r(x_l1.flip(dims=[1])).flip(dims=[1])
        for i in range(bs):
            x_l2[i, :N, :] = new_x1[i,:N,:][idx_pro_1[i].long(), :]
            x_l2[i, N:, :] = new_x1[i,N:,:]

        attn = 1

        x = x + x_l2
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder2(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder2, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx_emb_2, idx_pro_2, N, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, idx_emb_2, idx_pro_2, N, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, idx_emb_2, idx_pro_2, N, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, idx_emb_2, idx_pro_2, N, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


