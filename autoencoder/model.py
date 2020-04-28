# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn


class AutoEncoder(nn.Module):
    # Convolution Neural Network(CNN) 를 이용한 Autoencoder(오토인코더)
    def __init__(self,
                 input_dims,
                 encoder_filters,
                 encoder_conv_kernels,
                 encoder_conv_strides,
                 decoder_filters,
                 decoder_conv_kernels,
                 decoder_conv_strides,
                 latent_dim,
                 use_batch_norm,
                 use_dropout):
        super().__init__()
        self.encoder = Encoder(
            input_dims=input_dims,
            conv_filters=encoder_filters,
            conv_kernels=encoder_conv_kernels,
            conv_strides=encoder_conv_strides,
            latent_dim=latent_dim
        )

        self.decoder = Decoder(
            encoder_last_conv_dim=self.encoder.last_conv_dim,
            conv_filters=decoder_filters,
            conv_kernels=decoder_conv_kernels,
            conv_strides=decoder_conv_strides,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        return latent_space, x


class Encoder(nn.Module):
    def __init__(self,
                 input_dims,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_dim):
        super().__init__()
        # Convolution Neural Network 구성
        temp = []
        w, h, _ = input_dims
        for i, (in_channels, out_channels) in enumerate(zip(conv_filters[:-1],
                                                            conv_filters[1:])):
            temp.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=conv_kernels[i], stride=conv_strides[i]),
                nn.LeakyReLU()
            ])
            w = int((w - conv_kernels[i]) / conv_strides[i] + 1)
            h = int((h - conv_kernels[i]) / conv_strides[i] + 1)

        # 최종너비(s) * 최종높이(h) * 마지막 필터의 갯수
        self.last_dim = (w * h) * conv_filters[-1]
        self.last_conv_dim = (conv_filters[-1], w, h)

        self.latent_dim = latent_dim
        self.conv = nn.Sequential(*temp)
        self.fc = nn.Linear(self.last_dim, latent_dim, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        latent_space = self.fc(x)
        return latent_space


class Decoder(nn.Module):
    def __init__(self,
                 encoder_last_conv_dim,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_dim,
                 use_batch_norm,
                 use_dropout):
        super().__init__()
        self.encoder_last_conv_dim = encoder_last_conv_dim
        encoder_last_dim = int(np.prod(encoder_last_conv_dim))
        self.fc = nn.Linear(latent_dim, encoder_last_dim)

        # Transpose Convolution Neural Network 구성
        temp = []
        for i, (in_channels, out_channels) in enumerate(zip(conv_filters[:-1],
                                                            conv_filters[1:])):
            temp.append(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=conv_kernels[i], stride=conv_strides[i]),
            )

            if use_batch_norm:
                temp.append(nn.BatchNorm2d(out_channels))

            nn.LeakyReLU()

            if use_dropout:
                temp.append(nn.Dropout(0.5))

        self.conv = nn.Sequential(*temp)

    def forward(self, latent_space):
        x = self.fc(latent_space)
        x = torch.reshape(x, [-1, *self.encoder_last_conv_dim])
        x = self.conv(x)
        return x


