# Bsed on the work in https://github.com/eriklindernoren/Action-Recognition
# Python modules
import os
# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# local files
from .model_irse import IR_50
from .model_xception import GetPretrainedXception, Head


abspath = os.path.abspath(__file__)
__all__ = ['ConvLSTM', 't_sigmoid']


def t_sigmoid(x, t=1.):
    return 1./(1+torch.exp(-x/t))


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)

        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])
        last_layer = list(model.classifier.children())[-1]
        self.fc = nn.Linear(last_layer.in_features, latent_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
        return self.fc(x)


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class Xception(nn.Module):
    def __init__(self, latent_dim=1048, fine_tune=False):
        super(Xception, self).__init__()
        model = GetPretrainedXception()
        self.base = model.base
        self.h1 = Head(2048, latent_dim)
        self.fine_tune = fine_tune

    def forward(self, x):
        if self.fine_tune:
            x = self.base(x)
        else:
            with torch.no_grad():
                x = self.base(x)
        return self.h1(x)

    def set_ft(self, fine_tune=True):
        self.fine_tune = fine_tune


class IREncoder(nn.Module):
    def __init__(self, latent_dim, fine_tune=False):
        super(IREncoder, self).__init__()
        self.features = IR_50(input_size=(112, 122))
        path = '{}/../weights/face_evolve/backbone_ir50_ms1m_epoch120.pth'.format(
            abspath)
        self.features.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage))
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )
        self.fine_tune = fine_tune

    def forward(self, x):
        if self.fine_tune:
            x = self.features(x)
        else:
            with torch.no_grad():
                x = self.features(x)
        return self.fc(x)

    def set_ft(self, fine_tune=True):
        self.fine_tune = fine_tune


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


class ConvLSTM(nn.Module):
    """ Convolitional LSTM model for video predictions
    """

    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024,
            bidirectional=True, attention=True, encoder='VGG',
            calibrating=True, fine_tune=False
    ):
        """ Inintialization
        Parameters
        ----------
        num_classes : int
            Number of output classes
        latent_dim : int
            Latent dimension for embeddings fed into LSTMs
        lstm_layers : int
            Number of lstm layers to use in model
        hidden_dim : int
            Hidden kayer dimension in final prediction block
        bidirectional : bool
            Bi/Unidrectional switch
        attention : bool
            Attention block switch
        encoder : str
            Encoder architecture
        Returns
        -------
        """
        super(ConvLSTM, self).__init__()
        implemented_encoders = ['VGG', 'ResNet', 'IR', 'Xception']
        assert encoder in implemented_encoders, 'Selected encoder is missing'
        print('Building {} model'.format(encoder))
        if encoder == 'VGG':
            self.encoder = Encoder(latent_dim)
        if encoder == 'ResNet':
            self.encoder = ResNetEncoder(latent_dim)
        if encoder == 'IR':
            self.encoder = IREncoder(latent_dim, fine_tune=fine_tune)
        if encoder == 'Xception':
            self.encoder = Xception(latent_dim=latent_dim, fine_tune=fine_tune)

        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(
                2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            # nn.Softmax(dim=-1),
            # nn.Sigmoid()
        )
        self.attention = attention
        self.attention_layer = nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.calibrating = calibrating

    def forward(self, x):
        """ Forward pass
        Parameters
        ----------
        x : torch.tensor
            Tensor with video frames, expected size (bs, n_frames, c, h, w)
        Returns
        -------
        x : torch.tensor
            Processed torch data
        """
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(
                self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        x = self.output_layers(x)
        if self.calibrating is False:
            x = self.sigmoid(x)
        return x.view(x.shape[0])

        def set_ft(self, fine_tune=True):
            self.encoder.set_ft(fine_tune)
