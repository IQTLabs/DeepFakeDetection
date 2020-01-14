import gc
from pathlib import Path
from tqdm import tqdm
import skvideo
import skvideo.io
import pandas as pd

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms


__all__ = ['CreateOptim', 'save_checkpoint', 'preprocess_df']


def CreateOptim(parameters, lr=0.001, betas=(0.5, 0.999), factor=0.2,
                patience=5, threshold=1e-03,  eps=1e-08):
    """ Creates optimizer and associated learning rate scheduler for a model
    Paramaters
    ----------
    parameters : torch parameters
        Pytorch network parameters for associated optimzer and scheduler
    lr : float
        Learning rate for optimizer
    betas : 2-tuple(floats)
        Betas for optimizer
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
    optimizer = optim.Adam(parameters, lr=lr, betas=(0.5, 0.999))
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


def preprocess_df(df=None, mtcnn=None, path=None, outpath=None, n_seconds=10,
                  debug=False):
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
    n_seconds : int 
        Number fo seconds to load
    debug : bool
        Debug switch to test memory leak
    Returns
    -------
    faces_dataframe : pd.DataFrame
        Dataframe of preprocessed data
    """
    to_pil = transforms.ToPILImage()
    pbar = tqdm(total=len(df))
    faces_dataframe = []
    for idx in range(len(df)):
        pbar.update(1)
        entry = df.iloc[idx]
        filename = '{}/{}/{}'.format(path, entry['split'], entry['File'])
        dest = '{}/{}/{}/'.format(outpath, entry['split'], entry['File'])
        Path(dest).mkdir(parents=True, exist_ok=True)
        try:
            videodata = skvideo.io.vread(filename, num_frames=(n_seconds+1)*30)
        except:
            videodata = skvideo.io.vread(filename)
        f, h, w, c = videodata.shape
        # Temporary fix for large files, need to build propoer solution
        if h*w > (1080*1920):
            continue
        frames = [to_pil(x) for x in videodata[0::n_seconds]]
        faces = mtcnn(frames)
        n_frames = 0
        for ii, face in enumerate(faces):
            if n_frames == 30:
                break
            if face is None:
                continue
            imface = to_pil(face/2 + 0.5)
            imface.save('{}/frame_{}.jpeg'.format(dest, ii))
            del imface
            n_frames += 1
        this_entry = {'split': entry['split'], 'File': entry['File'],
                      'label': entry['label'], 'frames': n_frames}
        faces_dataframe.append(this_entry)
        del faces, frames, videodata, entry, this_entry
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if debug:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
    pbar.close()
    return pd.DataFrame(faces_dataframe)
