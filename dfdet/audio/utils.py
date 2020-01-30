import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dnn_models import MLP, flip
from .dnn_models import SincNet as CNN

__all__ = ['CreateSincNet', 'sample_settings', 'save_sincnet']


def str_to_bool(s):
    """  String to bool used to read config file
    Parameters
    ----------
    s : str
        String to convert
    Returns
    -------
    s : bool
        Boolean value of input string
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


def sample_settings(options):
    """ Load sample settings from the options
    Parameters
    ----------
    options : dict
        Dictionary with config settings
    Returns
    -------
    wlen : int
        Window length
    wshift : int
        Window start shift
    """
    wlen = int(options.fs*options.cw_len/1000.00)
    wshift = int(options.fs*options.cw_shift/1000.00)
    return wlen, wshift


def CreateSincNet(options):
    """ Create SincNet model from loaded configuration.  Predictions from
    model are produced by following pred = DNN1_net(DNN1_net(CNN_net(x)))
    Parameters
    ----------
    options : dict
        Dictionary with configuration loaded from file
    Returns
    -------
    CNN_net : nn.Module
        CNN block for SincNet
    DNN1_net : nn.Module
        First multilayer perceptron for feature (d-vector) extraction
    DNN2_net : nn.Module
        Second MLP for classification block
    """
    # [windowing]
    fs = int(options.fs)
    cw_len = int(options.cw_len)
    cw_shift = int(options.cw_shift)

    # [cnn]
    cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
    cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
    cnn_use_laynorm = list(
        map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm = list(
        map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cnn_act = list(map(str, options.cnn_act.split(',')))
    cnn_drop = list(map(float, options.cnn_drop.split(',')))

    # [dnn]
    fc_lay = list(map(int, options.fc_lay.split(',')))
    fc_drop = list(map(float, options.fc_drop.split(',')))
    fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
    fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
    fc_use_batchnorm = list(
        map(str_to_bool, options.fc_use_batchnorm.split(',')))
    fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    fc_act = list(map(str, options.fc_act.split(',')))

    # [class]
    class_lay = list(map(int, options.class_lay.split(',')))
    class_drop = list(map(float, options.class_drop.split(',')))
    class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
    class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
    class_use_batchnorm = list(
        map(str_to_bool, options.class_use_batchnorm.split(',')))
    class_use_laynorm = list(
        map(str_to_bool, options.class_use_laynorm.split(',')))
    class_act = list(map(str, options.class_act.split(',')))

    wlen, wshift = sample_settings(options)

    CNN_arch = {'input_dim': wlen,
                'fs': fs,
                'cnn_N_filt': cnn_N_filt,
                'cnn_len_filt': cnn_len_filt,
                'cnn_max_pool_len': cnn_max_pool_len,
                'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                'cnn_use_laynorm': cnn_use_laynorm,
                'cnn_use_batchnorm': cnn_use_batchnorm,
                'cnn_act': cnn_act,
                'cnn_drop': cnn_drop,
                }

    CNN_net = CNN(CNN_arch)

    DNN1_arch = {'input_dim': CNN_net.out_dim,
                 'fc_lay': fc_lay,
                 'fc_drop': fc_drop,
                 'fc_use_batchnorm': fc_use_batchnorm,
                 'fc_use_laynorm': fc_use_laynorm,
                 'fc_use_laynorm_inp': fc_use_laynorm_inp,
                 'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                 'fc_act': fc_act,
                 }

    DNN1_net = MLP(DNN1_arch)

    DNN2_arch = {'input_dim': fc_lay[-1],
                 'fc_lay': class_lay,
                 'fc_drop': class_drop,
                 'fc_use_batchnorm': class_use_batchnorm,
                 'fc_use_laynorm': class_use_laynorm,
                 'fc_use_laynorm_inp': class_use_laynorm_inp,
                 'fc_use_batchnorm_inp': class_use_batchnorm_inp,
                 'fc_act': class_act,
                 }

    DNN2_net = MLP(DNN2_arch)

    return CNN_net, DNN1_net, DNN2_net


def save_sincnet(CNN_net, DNN_1, DNN_2, description, filename='checkpoint.pth.tar'):
    """ Saves SincNet Model output = DNN_2(DNN_1(CNN_net(input)))
    Parameters
    ----------
    CNN_net : torch.Module
        First stage of SincNet
    DNN_1 : torch.Module
        Second stage of SincNet
    DNN_2 : torch.Module
        Third stage of SincNet
    description : str
        Descriptor for model checkpoint
    filename : str
        File name for save
    Returns
    -------
    """
    state = {
        'description': description,
        'CNN_net': CNN_net.state_dict(),
        'DNN_1': DNN_1.state_dict(),
        'DNN_2': DNN_2.state_dict()
    }
    torch.save(state, filename)
