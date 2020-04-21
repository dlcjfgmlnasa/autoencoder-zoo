# -*- coding:utf-8 -*-
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_dims,
                 latent_dim):
        super().__init__()
        dims = [input_dim] + encoder_dims + [latent_dim]
        temp = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            temp.append(nn.Linear(in_dim, out_dim, bias=True))
            temp.append(nn.LeakyReLU())
        self.fc = nn.Sequential(*temp)

    def forward(self, x):
        x = self.fc(x)
        return x
