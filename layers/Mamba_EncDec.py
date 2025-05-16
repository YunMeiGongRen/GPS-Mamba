import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer

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
        
        x_l1 = x[torch.arange(x.size(0), device=x.device)[:, None], idx_emb_1.long(), :]
        new_x1 = self.attention1(x_l1) + self.attention1_r(x_l1.flip(dims=[1])).flip(dims=[1])
        x_l2 = new_x1[torch.arange(x.size(0), device=x.device)[:, None], idx_pro_1.long(), :]
        x = torch.cat((x[:, :N, :] + x_l2, x[:, N:, :]), dim=1)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder1(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder1, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx_emb_1, idx_pro_1, N, attn_mask=None, tau=None, delta=None):

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x = attn_layer(x, idx_emb_1, idx_pro_1, N, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, idx_emb_1, idx_pro_1, N, tau=tau, delta=None)
 
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, idx_emb_1, idx_pro_1, N, attn_mask=attn_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        return x

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

        x_l1 = x[torch.arange(x.size(0), device=x.device)[:, None], idx_emb_1.long(), :]
        new_x1 = self.attention1(x_l1) + self.attention1_r(x_l1.flip(dims=[1])).flip(dims=[1])
        x_l2 = new_x1[torch.arange(x.size(0), device=x.device)[:, None], idx_pro_1.long(), :]
        x = torch.cat((x[:, :N, :] + x_l2, x[:, N:, :]), dim=1)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder2(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder2, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx_emb_2, idx_pro_2, N, attn_mask=None, tau=None, delta=None):
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x = attn_layer(x, idx_emb_2, idx_pro_2, N, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, idx_emb_2, idx_pro_2, N, tau=tau, delta=None)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, idx_emb_2, idx_pro_2, N, attn_mask=attn_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        return x


