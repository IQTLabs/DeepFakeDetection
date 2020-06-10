# Python modules
import os
# PyTorch modules
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model


abspath = os.path.abspath(__file__)


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(torch.nn.Module):
    def __init__(self, base, in_f=2048, out_f=1):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, out_f)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


def GetPretrainedXception(path='{}/../weights/xception/best_model.pth.tar'.format(abspath)):
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1])
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model = FCN(model, 2048)
    chpt = torch.load(path)
    model.load_state_dict(chpt['model'])
    return model
