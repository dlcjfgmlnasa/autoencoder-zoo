# -*- coding:utf-8 -*-
import os
import torch
import torchvision
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def mnist_train_dataloader(data_dir='./dataset', batch_size=32):
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
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def mnist_test_dataloader(data_dir='./dataset'):
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
    data_loader = DataLoader(dataset=dataset, shuffle=False)
    return data_loader


def latent_visualization(title, ckpt_dir, data_dir, figure_dir):
    figure_path = os.path.join(figure_dir, title+'.png')
    dataloader = mnist_test_dataloader(
        data_dir=data_dir
    )
    # model load : 학습된 모델 불러오기
    checkpoint = torch.load(ckpt_dir)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cpu()

    # inference : Autoencoder 추론
    features, labels = [], []
    with torch.no_grad():
        for data in dataloader:
            x, y = data
            x = x.cpu()
            y = y.item()

            latent, _ = model(x)
            latent = latent.squeeze(dim=0)
            latent = latent.tolist()
            features.append(latent)
            labels.append(y)

    # reduce dimensionality with t-sne : T-SNE 알고리즘 적용
    t_sne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
    t_sne_results = t_sne.fit_transform(features)
    t_sne_results_df = pd.DataFrame(data=t_sne_results, columns=['x', 'y'])
    t_sne_results_df['label'] = labels

    # visualization : 2차원 시각화
    plt.title(title)
    plt.scatter(x=t_sne_results_df['x'], y=t_sne_results_df['y'], c=t_sne_results_df['label'],
                cmap=plt.cm.get_cmap('jet', 10),
                s=2, alpha=0.7)
    plt.colorbar(ticks=range(10))
    plt.savefig(figure_path)
    plt.show()


