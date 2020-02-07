import yaml
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dfdet import *


parser = parser = argparse.ArgumentParser(description='Batch inference script')

parser.add_argument('--gpu', dest='gpu', default=0,
                    type=int, help='Target gpu')
parser.add_argument('--config', dest='config',
                    default='./config_files/training.yaml', type=str,
                    help='Config file with paths and MTCNN set-up')
parser.add_argument('--verbose', dest='verbose', default=False,
                    type=bool, help='Verbose switch')
parser.add_argument('--load', dest='chpt', default=None, type=str,
                    help='Checkpoint to resume training')


def train_test_split(df, fraction=0.8, random_state=200):
    df = df[df['frames'] >= 30]
    train = pd.concat([df[df['label'] == 1].sample(frac=fraction,
                                                   random_state=random_state),
                       df[df['label'] == 0].sample(frac=fraction,
                                                   random_state=random_state)])
    test = df.drop(train.index)
    return train.reindex(), test.reindex()


def paired_split(df, fraction=0.8, random_state=200):
    df = df[df['frames'] >= 30]
    all_df = pd.read_csv('../training_metadata.json')
    df.loc[df['label'] == 'REAL', 'label'] = 0
    df.loc[df['label'] == 'FAKE', 'label'] = 1
    test_fake = df[df['label'] == 1].sample(frac=1.0-fraction,
                                            random_state=random_state)
    originals = all_df.loc[all_df.File.isin(
        list(test_fake['File'])) == True, 'original']
    test_real = df[df.File.isin(list(originals)) == True]
    test = pd.concat([test_real, test_fake])
    train = df.drop(test.index)
    return train.reindex(), test.reindex()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    df = pd.read_csv('{}/faces_fixed_metadata.csv'.format(config['data_path']))
    if bool(config['paired_split']):
        train, test = paired_split(df, config['training_fraction'])
    else:
        train, test = train_test_split(df, config['training_fraction'])
    #
    if config['encoder'] == 'IR':
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    trainset = DFDC_Dataset(
        df=train, transform=transform, path=config['data_path'])
    trainloader = DataLoader(
        trainset, batch_size=config['batch_size'], shuffle=True,
        num_workers=16)
    #
    testset = DFDC_Dataset(
        df=test, transform=transform, path=config['data_path'])
    testloader = DataLoader(
        testset, batch_size=config['batch_size'], shuffle=False,
        num_workers=16)

    model = ConvLSTM(
        num_classes=1, lstm_layers=config['lstm_layers'],
        attention=config['attention'], encoder=config['encoder'],
        calibrating=False, fine_tune=config['fine_tune'])

    if args.chpt is not None:
        print('Loading file: {}'.format(args.chpt))
        chpt_file = torch.load(args.chpt)
        model.load_state_dict(chpt_file['model'])

    optim_, sched_ = CreateOptim(model.parameters(), lr=float(config['lr']),
                                 weight_decay=float(config['weight_decay']))

    losses = []
    averages = []
    train_dfd(model=model, dataloader=trainloader, testloader=testloader,
              optim=optim_, scheduler=sched_, criterion=nn.BCELoss(),
              losses=losses, averages=averages, n_epochs=config['n_epochs'],
              e_saves=config['e_saves'], device='cuda:{}'.format(args.gpu),
              verbose=args.verbose)
