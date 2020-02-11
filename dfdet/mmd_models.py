import torch
from torch import nn
import torch.nn.functional as F

from .video import LSTM, ConvLSTM
from .audio import CreateSincNet


__all__ = ['MultiModal_DeepFakeDetector']


class VideoStream(nn.Module):
    def __init__(
            self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024,
            bidirectional=True, attention=True, encoder='VGG', video_path=None,
            fine_tune=False
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
        super(VideoStream, self).__init__()
        model = ConvLSTM(num_classes=1, lstm_layers=lstm_layers,
                         attention=attention, encoder=encoder,
                         calibrating=True)
        if video_path is not None:
            chpt = torch.load(video_path)
            model.load_state_dict(chpt['model'])
        self.encoder = model.encoder
        del model
        self.fine_tune = fine_tune
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(
                2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            # nn.Softmax(dim=-1),
            # nn.Sigmoid()
        )
        self.attention = attention
        self.attention_layer = nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, 1)

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
        if self.fine_tune:
            x = self.encoder(x)
        else:
            with torch.no_grad():
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

        return x

    def set_ft(self, fine_tune=True):
        self.fine_tune = fine_tune


class MultiModal_DeepFakeDetector(nn.Module):
    def __init__(self, video_conf, audio_conf, audio_path=None,
                 video_path=None, fine_tune=False):
        super(MultiModal_DeepFakeDetector, self).__init__()

        self.video_stream = VideoStream(num_classes=1,
                                        lstm_layers=video_conf['lstm_layers'],
                                        attention=video_conf['attention'],
                                        encoder=video_conf['encoder'],
                                        video_path=video_path,
                                        fine_tune=fine_tune)

        self.CNN_net, self.DNN_1, DNN_2 = CreateSincNet(options=audio_conf)
        if audio_path is not None:
            self.load_audio_weights(audio_path)

        self.DNN_2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.vote = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1)
        )

        self.fine_tune = fine_tune

    def forward(self, video, audio):
        if self.fine_tune:
            audio = self.DNN_1(self.CNN_net(audio))
        else:
            with torch.no_grad():
                audio = self.DNN_1(self.CNN_net(audio))
        audio = self.DNN_2(audio)
        video = self.video_stream(video)
        x = torch.cat([video, audio], dim=1)
        return F.sigmoid(self.vote(x))

    def load_audio_weights(self, path=''):
        chpt = torch.load(path)
        self.CNN_net.load_state_dict(chpt['CNN_net'])
        self.DNN_1.load_state_dict(chpt['DNN_1'])
        del chpt

    def set_ft(self, fine_tune=True):
        self.fine_tune = fine_tune
