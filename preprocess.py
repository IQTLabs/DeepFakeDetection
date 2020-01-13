import yaml
import argparse
import pandas as pd

import torch

from facenet_pytorch import MTCNN
from src import preprocess_df

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
