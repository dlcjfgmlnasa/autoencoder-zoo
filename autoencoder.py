# -*- coding:utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from model import encoder, decoder
from utils import mnist_train_dataloader, latent_visualization
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.path.join('.', 'dataset'))
    parser.add_argument('--model_path', type=str, default=os.path.join('.', 'save_model'))
    parser.add_argument('--figure_path', type=str, default=os.path.join('.', 'fig'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--encoder_dims', type=list, default=[400, 200, 100])
    parser.add_argument('--decoder_dims', type=list, default=[200, 400, 100])
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--writer_point_step', type=int, default=100)
    parser.add_argument('--save_point_step', type=int, default=1000)
    return parser.parse_args()


class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_dims,
                 decoder_dims,
                 latent_dim):
        super().__init__()
        input_dim = input_dim
        self.encoder = encoder.Encoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            latent_dim=latent_dim)

        self.decoder = decoder.Decoder(
            latent_dim=latent_dim,
            decoder_dims=decoder_dims,
            output_dim=input_dim)

    def forward(self, x):
        x = torch.reshape(x, [x.shape[0], -1])
        latent = self.encoder(x)
        y = self.decoder(latent)
        return latent, y


def train(arguments):
    model_path = os.path.join(arguments.model_path, 'autoencoder')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataloader = mnist_train_dataloader(
        path=arguments.dataset_path,
        batch_size=arguments.batch_size
    )
    writer = SummaryWriter(
        logdir='./logs/autoencoder',
    )

    model = AutoEncoder(
        input_dim=28*28*1,
        encoder_dims=arguments.encoder_dims,
        decoder_dims=arguments.decoder_dims,
        latent_dim=arguments.latent_dim).to(device)

    criterion = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=arguments.learning_rate)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=arguments.learning_rate)

    count = 0
    total_it = 0
    step_total_loss = 0

    writer.add_hparams({
        'epochs': arguments.epochs,
        'batch_size': arguments.batch_size,
        'learning_rate': arguments.learning_rate,
        'encoder_dims': arguments.encoder_dims,
        'decoder_dims': arguments.decoder_dims,
        'latent_dim': arguments.latent_dim
    })

    for epoch in range(arguments.epochs):
        for it, data in enumerate(train_dataloader):
            x, _ = data
            x = x.to(device)

            # autoencoder model
            l, out = model(x)

            # optimizer & gradient
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = criterion(input=out, target=torch.reshape(x, [x.shape[0], -1]))
            step_total_loss += loss.item()
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_it += 1
            count += 1

            if total_it % arguments.writer_point_step == 0:
                step_total_loss /= count

                # console writer
                print('[Total Iter] : {0:05d}it [Epoch] : {1:03d}s [Loss] : {2:.05f}'.format(
                    total_it, epoch, step_total_loss)
                )

                # tensorboard writer
                writer.add_scalar('loss', step_total_loss, total_it)
                step_total_loss, count = 0, 0

            if total_it % arguments.save_point_step == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'loss': step_total_loss,
                    'model': model
                }, os.path.join(model_path, 'step_{0:05d}_epoch_{1:03d}_batch_size_{2:03d}_lr_{3:.03f}.pth'.format(
                    total_it, epoch, arguments.batch_size, arguments.learning_rate)))


if __name__ == '__main__':
    args = get_args()

    # train(args)
    latent_visualization(
        title='autoencoder',
        dataloader_path=os.path.join('.', 'dataset'),
        model_path=os.path.join('.', 'save_model', 'autoencoder',
                                'step_12000_epoch_019_batch_size_100_lr_0.001.pth'),
        figure_path=args.figure_path
    )




