import torch
from torch import nn

__all__ = ['FFTHead']


class FFTHead(torch.nn.Module):
    def __init__(self, in_f, out_f, latent_dim):
        super(FFTHead, self).__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_f),
            nn.Dropout(0.75),
            nn.Linear(in_f, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(0.75)
        )

        self.classifier = nn.Linear(latent_dim, out_f)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
