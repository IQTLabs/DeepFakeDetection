import gc
from pathlib import Path
from tqdm import tqdm
import skvideo
import skvideo.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from moviepy.editor import *


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms


__all__ = ['CreateOptim', 'save_checkpoint',
           'preprocess_df', 'preprocess_df_audio']


def plot_losses(train=[], test=[], path=''):
    """ Plotting function for training/val losses
    Parameters
    ----------
    train : list
        Training losses over training
    test : list
        Test losses over training
    path : str
        Path for output plot
    """
    epochs = [x for x in range(len(train))]
    fig = plt.figure(figsize=(5, 5))
    sns.lineplot(x=epochs, y=train, label='Train')
    sns.lineplot(x=epochs, y=test, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('{}/Loss.jpeg'.format(path))


def CreateOptim(parameters, lr=0.001, betas=(0.5, 0.999), weight_decay=0,
                factor=0.2, patience=5, threshold=1e-03,  eps=1e-08):
    """ Creates optimizer and associated learning rate scheduler for a model
    Paramaters
    ----------
    parameters : torch parameters
        Pytorch network parameters for associated optimzer and scheduler
    lr : float
        Learning rate for optimizer
    betas : 2-tuple(floats)
        Betas for optimizer
    weight_decay : float
        Weight decay for optimizer regularization
    factor : float
        Factor by which to reduce learning rate on Plateau
    patience : int
        Patience for learning rate scheduler
    Returns
    -------
    optimizer : torch.optim
        optimizer for model
    scheduler : ReduceLROnPlateau
        scheduler for optimizer
    """
    optimizer = optim.Adam(parameters, lr=lr, betas=(
        0.5, 0.999), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=patience,
        threshold=threshold, eps=eps, verbose=True)
    return optimizer, scheduler


def save_checkpoint(model, description, filename='checkpoint.pth.tar'):
    """ Saves input state dict to file
    Parameters
    ----------
    state : dict
        State dict to save. Can include parameters from model, optimizer, etc.
        as well as any other elements.
    is_best : bool
        If true will save current state dict to a second location
    filename : str
        File name for save
    Returns
    -------
    """
    state = {
        'architecture': str(model),
        'description': description,
        'model': model.state_dict()
    }
    torch.save(state, filename)


def preprocess_df(df=None, mtcnn=None, path=None, outpath=None,
                  target_n_frames=60, frame_rate=10, mini_batch=15,
                  n_seconds=5, debug=False):
    """ Preprocessing script for deep fake challenge.  Subsamples, videos,
    isolates faces and saves frames.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with video metadata
    mtcnn : torch.Module
        Facial detection module from facenet-python (https://github.com/timesler/facenet-pytorch)
    path : str
        Path to directory with DFDC data
    outpath : str
        Destination for preprocessed frames
    target_n_frames : int
        Target number of frames to extract
    frame_rate : int
        Number of frames per second to process
    mini_batch : str
        Mini batch size for preprocessing steps (protects against memory overflow)
    n_seconds : int
        Numbr of seconds to load video (speed up optimization)
    debug : bool
        Debug switch to test memory leak
    Returns
    -------
    faces_dataframe : pd.DataFrame
        Dataframe of preprocessed data
    """
    def split(my_list, n):
        """ Splits list into lists of length n
        Paramaters
        ----------
        my_list : list
            List to subdivide
        n : int
            Max length of desired sub-lists
        Returns
        final : list
            List of sub-lists of length n
        """
        final = [my_list[i * n:(i + 1) * n]
                 for i in range((len(my_list) + n - 1) // n)]
        return final

    def process_min_batch(batch=None, start_index=0):
        """ Pre-process and save a mini_batch of frames
        Parameters
        ----------
        batch : list(torch.tensor)
            List with frames to preprocess
        start_index : int
            Number fo previously saved frames in the video
        Returns
        -------
        end_index : int
            Number of saved frames at end of this mini-batch
        """
        with torch.no_grad():
            faces, probs = mtcnn(batch, return_prob=True)
        saved_frames = 0
        faces_probs = []
        for ii, face in enumerate(faces):
            if face is None or probs[ii] < 0.95:
                continue
            faces_probs.append(probs[ii])
            imface = to_pil(face/2 + 0.5)
            imface.save('{}/frame_{}.png'.format(dest,
                                                 saved_frames+start_index))
            del imface
            saved_frames += 1
        del faces
        return saved_frames, faces_probs

    frame_skip = 30//frame_rate
    to_pil = transforms.ToPILImage()
    pbar = tqdm(total=len(df))
    faces_dataframe = []
    for idx in range(len(df)):
        pbar.update(1)
        entry = df.iloc[idx]
        this_entry = {'split': entry['split'], 'File': entry['File'],
                      'label': entry['label'], 'frames': 0,
                      'probabilites': []}
        try:
            filename = '{}/{}/{}'.format(path, entry['split'], entry['File'])
            dest = '{}/{}/{}/'.format(outpath, entry['split'], entry['File'])
            Path(dest).mkdir(parents=True, exist_ok=True)
            try:
                videodata = skvideo.io.vread(filename, (n_seconds)*30)
            except RuntimeError:
                videodata = skvideo.io.vread(filename)
            except:
                continue
            frames = [to_pil(x) for x in videodata[0::frame_skip]]
            frames_batches = split(frames, mini_batch)
            n_frames = 0
            probabilities = []
            for batch in frames_batches:
                if n_frames >= target_n_frames:
                    break
                t_frames, t_probs = process_min_batch(batch, n_frames)
                n_frames += t_frames
                probabilities += t_probs

            this_entry['frames'] = n_frames
            this_entry['probabilities'] = probabilities
            del frames, videodata
        except:
            pass
        faces_dataframe.append(this_entry)
        del entry, this_entry
        #
        if debug:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
    pbar.close()
    return pd.DataFrame(faces_dataframe)


def preprocess_df_audio(df=None, path=None, outpath=None, fps=16000):
    """ Preprocessing to extact audio from videos
    Parameters
    ----------
    df : pd.DataFrame
        Metadata for files to pre-process
    path : str
        Path to source data
    outpath : str
        Path for output prpeorcessed data
    fps : 
        Framerate for data loading/saving
    Returns
    audio_dataframe : pd.Dataframe
        Extracted audio meta data
    """
    audio_dataframe = []
    pbar = tqdm(total=len(df))
    for idx in range(len(df)):
        pbar.update(1)
        entry = df.iloc[idx]
        this_entry = {'split': entry['split'], 'File': entry['File'],
                      'label': entry['label'], 'wlen': 0}
        try:
            filename = '{}/{}/{}'.format(path, entry['split'], entry['File'])
            dest = '{}/{}'.format(outpath, entry['split'])
            Path('{}'.format(dest)).mkdir(parents=True, exist_ok=True)
            audio = AudioFileClip(filename, fps=16000)
            audio_array = audio.to_soundarray()
            this_entry['wlen'] = audio_array.shape[0]
            np.save(file='{}/{}'.format(dest, entry['File']), arr=audio_array)
        except:
            pass
        audio_dataframe.append(this_entry)
    return pd.DataFrame(audio_dataframe)
