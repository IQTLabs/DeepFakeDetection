import os
import sys
import gc
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import imageio
import skvideo
import skvideo.io

import torch
import torchvision
from torchvision import transforms

from facenet_pytorch import MTCNN


parser = parser = argparse.ArgumentParser(description='Batch inference script')

parser.add_argument('--gpu', dest='gpu', default=0,
                    type=int, help='Target gpu')
parser.add_argument('--config', dest='config',
                    default='./config_files/preprocess.yaml', type=str,
                    help='Config file with paths and MTCNN set-up')
parser.add_argument('--df', dest='df', default=None,
                    type=str, help='Dataframe location')
parser.add_argument('--seconds', dest='seconds', default=30,
                    type=int, help='Number of seconds to sample from')
parser.add_argument('--debug', dest='debug', default=False,
                    type=bool, help='Debug mode switch')


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
            continue
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
                      'label': entry['label'], 'frames': len(faces)}
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


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.df is not None, 'Need to specify metadata file'
    with open(args.config) as f:
        config = yaml.load(f)
    device = torch.device('cuda:{}'.format(args.gpu))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.df)
    path = config['data_path']
    mt = config['mtcnn']
    mtcnn = MTCNN(
        image_size=mt['image_size'], margin=mt['margin'],
        min_face_size=mt['min_face_size'], thresholds=mt['thresholds'],
        factor=mt['factor'], post_process=mt['post_process'],
        device=device
    )
    mtcnn.eval()
    faces_dataframe = preprocess_df(df=df, mtcnn=mtcnn, path=path,
                                    outpath=config['out_path'],
                                    n_seconds=args.seconds, debug=args.debug)
    faces_dataframe.to_csv('{}/faces_metadata.csv'.format(config['out_path']))
