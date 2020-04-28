# -*- coding:utf-8 -*-
import os
import argparse
import torch.nn as nn
import torch.optim as opt
from model import AutoEncoder
from utils import *
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=[
        'train', 'latent_visualization', 'image_generator'
    ])
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'celeb'])
    parser.add_argument('--encoder_filters', type=list, default=[1, 32, 32, 64])
    parser.add_argument('--encoder_conv_kernels', type=list, default=[3, 3, 3, 3])
    parser.add_argument('--encoder_conv_strides', type=list, default=[1, 1, 1, 1])
    parser.add_argument('--decoder_filters', type=list, default=[64, 32, 32, 1])
    parser.add_argument('--decoder_conv_kernels', type=list, default=[3, 3, 3, 3])
    parser.add_argument('--decoder_conv_strides', type=list, default=[1, 1, 1, 1])
    parser.add_argument('--use_batch_norm', type=bool, default=True)
    parser.add_argument('--use_dropout', type=bool, default=True)
    parser.add_argument('--latent_dim', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'dataset'))
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'log'))
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join('.', 'ckpt'))
    parser.add_argument('--ckpt_name', type=str, default='step_09380_batch_size_064_lr_0.00010.pth')
    parser.add_argument('--figure_dir', type=str, default=os.path.join('.', 'figure'))

    parser.add_argument('--print_step_point', type=int, default=20)
    parser.add_argument('--save_step_point', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, arguments):
        self.arguments = arguments
        input_dims = 28, 28, 1

        self.ae = AutoEncoder(
            input_dims=input_dims,
            encoder_filters=arguments.encoder_filters,
            encoder_conv_kernels=arguments.encoder_conv_kernels,
            encoder_conv_strides=arguments.encoder_conv_strides,
            decoder_filters=arguments.decoder_filters,
            decoder_conv_kernels=arguments.decoder_conv_kernels,
            decoder_conv_strides=arguments.decoder_conv_strides,
            latent_dim=arguments.latent_dim,
            use_batch_norm=arguments.use_batch_norm,
            use_dropout=arguments.use_dropout
        ).to(device)
        self.ae.train()

        self.criterion = nn.MSELoss()
        self.encoder_optimizer = opt.Adam(params=self.ae.encoder.parameters(), lr=arguments.learning_rate)
        self.decoder_optimizer = opt.Adam(params=self.ae.decoder.parameters(), lr=arguments.learning_rate)
        self.writer = SummaryWriter(logdir=os.path.join(self.arguments.log_dir, self.arguments.data),
                                    comment='epoch_{0:03d}_batch_size_{1:03d}_lr_{2:.03f}'.format(
                                        self.arguments.epochs-1,
                                        self.arguments.batch_size,
                                        self.arguments.learning_rate
                                    ))

    def train(self):
        step = 0
        for epoch in range(self.arguments.epochs):
            for data in self.train_dataloader():
                x, _ = data
                x = x.to(device)

                _, out = self.ae(x)

                # Optimizer & Backward
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                loss = self.criterion(input=out, target=x)
                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                # Console & Tensorboard Log 출력
                if step % self.arguments.print_step_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:06d}  [Loss]: {2:.05f}'.format(
                        epoch, step, loss.item()
                    ))
                    self.writer.add_scalar('loss', loss.item())

                # 모델 저장
                if step % self.arguments.save_step_point == 0:
                    ckpt_dir = os.path.join(
                            self.arguments.ckpt_dir,
                            self.arguments.data,
                            'step_{0:05d}_batch_size_{1:03d}_lr_{2:.05f}.pth'.format(
                                step, self.arguments.batch_size, self.arguments.learning_rate))
                    model_save(
                        model=self.ae,
                        encoder_optimizer=self.encoder_optimizer,
                        decoder_optimizer=self.decoder_optimizer,
                        loss=loss.item(),
                        latent_dim=self.arguments.latent_dim,
                        ckpt_dir=ckpt_dir
                    )
                    print('save model \t => ', ckpt_dir)

                step += 1

        # 학습 후 마지막 결과 저장
        ckpt_dir = os.path.join(
            self.arguments.ckpt_dir,
            self.arguments.data,
            'step_{0:05d}_batch_size_{1:03d}_lr_{2:.05f}.pth'.format(
                step, self.arguments.batch_size, self.arguments.learning_rate))

        model_save(
            model=self.ae,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            loss=loss.item(),
            latent_dim=self.arguments.latent_dim,
            ckpt_dir=ckpt_dir
        )
        print('save model \t => ', ckpt_dir)

    def train_dataloader(self):
        if self.arguments.data == 'mnist':
            dataloader = mnist_train_dataloader(
                data_dir=os.path.join(args.data_dir, args.data),
                batch_size=self.arguments.batch_size
            )
            return dataloader

    def get_input_dims(self):
        if self.arguments.data == 'mnist':
            return 28, 28, 1


if __name__ == '__main__':
    args = get_args()

    # Autoencoder (오토인코더) 학습
    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()

    # Latent space (잠재 공간) 시각화
    if args.mode == 'latent_visualization':
        latent_visualization(
            data_dir=args.data_dir,
            figure_dir=args.figure_dir,
            figure_title='autoencoder_{}_{}_latent_dim'.format(args.data, args.latent_dim),
            ckpt_dir=args.ckpt_dir,
            ckpt_name=args.ckpt_name,
            data=args.data
        )

    # 이미지 생성
    if args.mode == 'image_generator':
        image_generator(
            figure_dir=args.figure_dir,
            figure_title='autoencoder_{}_image_sample'.format(args.data),
            ckpt_dir=args.ckpt_dir,
            ckpt_name=args.ckpt_name,
            sample_uniform_max=5,
            sample_uniform_min=-5
        )