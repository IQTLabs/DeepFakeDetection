import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['DFDC_Dataset', 'DFDC_MD_Dataset']


def fix_labels(df=None):
    df.loc[df['label'] == 'FAKE', 'label'] = 0
    df.loc[df['label'] == 'REAL', 'label'] = 1


class DFDC_Dataset(Dataset):
    """ DeepFake detection dataset
    """

    def __init__(self, df=None, transform=None, path='', frames=30,
                 stochastic=True):
        """ Dataset initialization
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with preprocessed data
        transform : torchvision.transforms
            Transformation operations for loaded images
        path : str
            Path to folder with the data
        frames : int
            Frames to load per video
        """
        assert df is not None, 'Missing dataframe for data'
        self.frames = frames
        self.df = df[df['frames'] >= frames]
        self.path = path
        self.stochastic = stochastic
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        """ Len of dataset
        Parameters
        ----------
        Returns
        -------
        length : int
            Dataset length
        """
        return len(self.df)

    def __getitem__(self, idx):
        """ Return dataset item
        Parameters
        ----------
        idx : int
            Item to retrieve
        Returns
        -------
        frames : torch.tensor
            Torch tensor with video frames size (1, n_frames, 3, h, w)
        lbls : torch.tensor
            Tensor with video lables (1=real, 0=fake)
        """
        entry = self.df.iloc[idx]
        dest = '{}/{}/{}/'.format(self.path, entry['split'], entry['File'])
        path, dirs, files = next(os.walk(dest))
        frames = []
        nframe = 0
        if self.stochastic:
            if entry['frames'] == 30:
                start = 61
            else:
                start = np.random.randint(91-entry['frames'], 61)

        while len(frames) < self.frames:
            f = '{}/frame_{}.png'.format(dest, start+nframe)
            if os.path.isfile(f):
                frames.append(self.transform(Image.open(f)))
            nframe += 1
            if nframe > 1000:
                print('Something terrible ', dest)
                break

        return torch.stack(frames), entry['label']


class DFDC_MD_Dataset(Dataset):
    """ DeepFake detection dataset
    """

    def __init__(self, df=None, transform=None, path='', audio_path='',
                 frames=30, wlen=0):
        """ Dataset initialization
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with preprocessed data
        transform : torchvision.transforms
            Transformation operations for loaded images
        path : str
            Path to folder with the data
        frames : int
            Frames to load per video
        """
        assert df is not None, 'Missing dataframe for data'
        self.frames = frames
        self.df = df[df['frames'] >= frames]
        self.path = path
        self.audio_path = audio_path
        self.wlen = wlen
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        """ Len of dataset
        Parameters
        ----------
        Returns
        -------
        length : int
            Dataset length
        """
        return len(self.df)

    def __getitem__(self, idx):
        """ Return dataset item
        Parameters
        ----------
        idx : int
            Item to retrieve
        Returns
        -------
        frames : torch.tensor
            Torch tensor with video frames size (1, n_frames, 3, h, w)
        lbls : torch.tensor
            Tensor with video lables (1=real, 0=fake)
        """
        entry = self.df.iloc[idx]
        dest = '{}/{}/{}/'.format(self.path, entry['split'], entry['File'])
        path, dirs, files = next(os.walk(dest))
        frames = []
        nframe = 0
        while len(frames) < self.frames:
            f = '{}/frame_{}.png'.format(dest, nframe)
            if os.path.isfile(f):
                frames.append(self.transform(Image.open(f)))
            nframe += 1
            if nframe > 1000:
                print('Something terrible ', dest)
                break

        audio_dest = '{}/{}/{}.npy'.format(self.audio_path,
                                           entry['split'], entry['File'])
        signal = np.load(audio_dest)
        instance = signal[0:self.wlen, 0]

        return torch.stack(frames), instance, entry['label']
