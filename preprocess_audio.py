import yaml
import argparse
import pandas as pd

from dfdet import preprocess_df_audio

parser = parser = argparse.ArgumentParser(description='Batch inference script')


parser.add_argument('--config', dest='config',
                    default='./config_files/preprocess_audio.yaml', type=str,
                    help='Config file with paths and MTCNN set-up')
parser.add_argument('--df', dest='df', default=None,
                    type=str, help='Dataframe location')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.df is not None, 'Need to specify metadata file'
    with open(args.config) as f:
        config = yaml.load(f)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.df)
    path = config['data_path']

    audio_dataframe = preprocess_df_audio(df=df, path=path,
                                          outpath=config['out_path'],
                                          fps=config['fps'])

    audio_dataframe.to_csv('{}/audio_metadata.csv'.format(config['out_path']))
