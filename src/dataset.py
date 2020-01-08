import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['DFDC_Dataset']


def fix_labels(df=None):
    df.loc[df['label'] == 'FAKE', 'label'] = 0
    df.loc[df['label'] == 'REAL', 'label'] = 1


class DFDC_Dataset(Dataset):
    def __init__(self, df=None, transform=None, frames=30):
        assert df is not None, 'Missing dataframe for data'
        self.frames = frames
        self.df = df[df['frames'] >= frames]
        self.path = '/home/mlomnitz/Documents/Data/deepfake-detection/dfdc_train_subset_faces'
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        dest = '{}/{}/{}/'.format(self.path, entry['split'], entry['File'])
        path, dirs, files = next(os.walk(dest))
        frames = []
        nframe = 0
        while len(frames) < self.frames:
            f = '{}/frame_{}.jpeg'.format(dest, nframe)
            if os.path.isfile(f):
                frames.append(self.transform(Image.open(f)))
            nframe += 1
            if nframe > 1000:
                print('Something terrible')
                break

        return torch.stack(frames), entry['label']
