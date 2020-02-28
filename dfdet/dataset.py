import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize
from albumentations.augmentations.transforms import RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression

__all__ = ['DFDC_Dataset', 'DFDC_MD_Dataset']


def fix_labels(df=None):
    df.loc[df['label'] == 'FAKE', 'label'] = 0
    df.loc[df['label'] == 'REAL', 'label'] = 1


class DFDC_Dataset(Dataset):
    """ DeepFake detection dataset
    """

    def __init__(self, df=None, size=150, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], augment=True, frames=30,
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
        self.df = df[df['nframes'] >= frames]
        self.stochastic = stochastic
        addtl_img = {}
        for idx in range(frames):
            addtl_img['image{}'.format(idx)] = 'image'
        if augment:
            self.transform = albumentations.Compose([
                ShiftScaleRotate(p=0.3, scale_limit=0.25,
                                 border_mode=1, rotate_limit=15),
                HorizontalFlip(p=0.2),
                RandomBrightnessContrast(
                    p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                MotionBlur(p=.2),
                GaussNoise(p=.2),
                JpegCompression(p=.2, quality_lower=50),
                Normalize(mean, std)
            ], additional_targets=addtl_img)
        else:
            self.transform = albumentations.Compose([
                Normalize(mean, std)
            ])
        self.resize = transforms.Resize((size, size))

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
        if self.stochastic:
            start = np.random.randint(0, entry['nframes']-self.frames)
        to_pass = {}
        for idx in range(self.frames):
            f = '{}/frame_{}.png'.format(entry['File'], start+idx)
            if idx == 0:
                to_pass['image'] = np.array(self.resize(Image.open(f)))
            else:
                to_pass['image{}'.format(idx)] = np.array(
                    self.resize(Image.open(f)))
        frames = self.transform(**to_pass)
        frames = np.stack([x for x in frames.values()])
        return torch.from_numpy(np.rollaxis(frames, 3, 1)), entry['label']


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
