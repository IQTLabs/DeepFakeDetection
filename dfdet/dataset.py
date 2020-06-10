from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize
from albumentations.augmentations.transforms import RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression

from .fft import *


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
            if entry['nframes']-self.frames <= 0:
                start = 0
            else:
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

    def __init__(self, df=None, size=150, size_gs=224,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 augment=True, frames=15, wlen=0, stochastic=True):
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
        self.df = df[(df['nframes'] >= frames) & (df['wlen'] > wlen)]

        self.wlen = wlen
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
            self.transform_gs = albumentations.Compose([
                ShiftScaleRotate(p=0.3, scale_limit=0.25,
                                 border_mode=1, rotate_limit=15),
                HorizontalFlip(p=0.2),
                RandomBrightnessContrast(
                    p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                MotionBlur(p=.2),
                GaussNoise(p=.2),
                JpegCompression(p=.2, quality_lower=50)
            ], additional_targets=addtl_img)
        else:
            self.transform = albumentations.Compose([
                Normalize(mean, std)
            ])
            self.transform_gs = None

        self.resize = transforms.Resize((size, size))
        self.resize_gs = transforms.Compose([
            transforms.Resize((size_gs, size_gs)),
            transforms.Grayscale()])

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
            if entry['nframes']-self.frames <= 0:  # frame start
                start = 0
            else:
                start = np.random.randint(0, entry['nframes']-self.frames)

            if entry['wlen']-self.wlen <= 0:
                audio_start = 0
            else:
                audio_start = np.random.randint(
                    0, int(entry['wlen'])-self.wlen)

        to_pass = {}
        to_pass_gs = {}
        for idx in range(self.frames):
            f = '{}/frame_{}.png'.format(entry['File'], start+idx)
            if idx == 0:
                img = Image.open(f)
                to_pass['image'] = np.array(self.resize(img))
                to_pass_gs['image'] = np.array(self.resize_gs(img))
            else:
                img = Image.open(f)
                to_pass['image{}'.format(idx)] = np.array(self.resize(img))
                to_pass_gs['image{}'.format(idx)] = np.array(
                    self.resize_gs(img))
        frames = self.transform(**to_pass)
        if self.transform_gs is not None:
            frames_gs = self.transform_gs(**to_pass_gs)
        else:
            frames_gs = to_pass_gs
        #
        frames = np.stack([x for x in frames.values()])
        spectrum = np.stack([getPoweSpectrum(x) for x in frames_gs.values()])

        signal = np.load(entry['Audio'])
        instance = signal[audio_start:audio_start+self.wlen, 0]

        return torch.from_numpy(np.rollaxis(frames, 3, 1)), instance, spectrum, entry['label']
