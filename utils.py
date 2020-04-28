# -*- coding:utf-8 -*-
import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm


def mnist_train_dataloader(data_dir=os.path.join('./dataset', 'mnist'),
                           batch_size=32):
    # MNIST train dataset & dataloader
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True,
    )
    dataloder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloder


def mnist_test_dataloader(data_dir=os.path.join('./dataset', 'mnist')):
    # MNIST test dataset & dataloader
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True,
    )
    dataloder = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    return dataloder


def model_save(model: torch.nn.Module,
               encoder_optimizer: torch.optim,
               decoder_optimizer: torch.optim,
               loss,
               latent_dim,
               ckpt_dir):

    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': loss,
        'latent_dim': latent_dim,
        'model': model
    }, ckpt_dir)


def latent_visualization(data_dir, data,
                         figure_dir, figure_title,
                         ckpt_dir, ckpt_name):
    # 그림파일 디렉토리
    figure_dir = os.path.join(figure_dir, figure_title+'.png')

    # model load : 학습된 모델 불러오기
    ckpt_dir = os.path.join(ckpt_dir, ckpt_name)
    checkpoint = torch.load(ckpt_dir)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cpu()

    # Dataset & Dataloader
    dataloader = None
    if data == 'mnist':
        dataloader = mnist_test_dataloader(
            data_dir=os.path.join(data_dir, data)
        )

    # inference : Autoencoder 추론
    features, labels = [], []
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Autoencoder Inference'):
            x, y = data
            x = x.cpu()
            y = y.tolist()

            latent_space, _ = model(x)
            latent_space = latent_space.squeeze(dim=0)
            latent_space = latent_space.tolist()
            features.extend(latent_space)
            labels.extend(y)

    # reduce dimensionality with t-sne : T-SNE 알고리즘 적용
    t_sne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
    t_sne_results = t_sne.fit_transform(features)
    t_sne_results_df = pd.DataFrame(data=t_sne_results, columns=['x', 'y'])
    t_sne_results_df['label'] = labels

    # visualization : 2차원 시각화
    plt.title(figure_title)
    plt.scatter(x=t_sne_results_df['x'], y=t_sne_results_df['y'], c=t_sne_results_df['label'],
                cmap=plt.cm.get_cmap('jet', 10),
                s=2, alpha=0.7)
    plt.colorbar(ticks=range(10))
    plt.savefig(figure_dir)
    plt.show()


def image_generator(figure_dir, figure_title,
                    ckpt_dir, ckpt_name,
                    sample_uniform_max, sample_uniform_min):
    # 그림파일 디렉토리
    figure_dir = os.path.join(figure_dir, figure_title)

    # 이미지 갯수
    image_count = 25

    # model load : 학습된 모델 불러오기
    ckpt_dir = os.path.join(ckpt_dir, ckpt_name)
    checkpoint = torch.load(ckpt_dir)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cpu()

    # sample 생성
    latent_dim = checkpoint['latent_dim']
    sample = np.random.uniform(
        low=sample_uniform_min,
        high=sample_uniform_max,
        size=(image_count, latent_dim)
    )
    sample = torch.tensor(sample, dtype=torch.float32)

    # Autoencoder(오토인코더) 이미지 생성
    with torch.no_grad():
        images = model.decoder(sample)
        images = images.numpy()

        # 이미지 생성
        fig = plt.figure()
        plt.title(figure_title)
        plt.axis('off')

        for i in range(image_count):
            subplot = fig.add_subplot(5, 5, i + 1)
            image = np.reshape(images[i], (28, 28))
            subplot.axis('off')
            subplot.imshow(image)

        plt.tight_layout()
        plt.savefig(figure_dir)
        plt.show()
