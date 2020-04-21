# -*- coding:utf-8 -*-
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 decoder_dims,
                 output_dim):
        super().__init__()
        dims = [latent_dim] + decoder_dims
        temp = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            temp.append(nn.Linear(in_dim, out_dim, bias=True))
            temp.append(nn.LeakyReLU())
        temp.append(nn.Linear(dims[-1], output_dim, bias=True))
        self.fc = nn.Sequential(*temp)

    def forward(self, x):
        x = self.fc(x)
        return x
