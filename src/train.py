from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ['train']


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        """ Initialize objects and reset for safety
        Parameters
        ----------
        Returns
        -------
        """
        self.reset()

    def reset(self):
        """ Resets the meter values if being re-used
        Parameters
        ----------
        Returns
        -------
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update meter values give current value and batchsize
        Parameters
        ----------
        val : float
            Value fo metric being tracked
        n : int
            Batch size
        Returns
        -------
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar = tqdm(total=len(dataloader))
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for idx, batch in enumerate(dataloader):
        frames, lbls = batch
        frames, lbls = frames.to(device), lbls.to(device)
        with torch.no_grad():
            model.lstm.reset_hidden_state()
            predictions = model(frames)
        correct = (predictions.detach().argmax(
            dim=1) == lbls).sum().cpu().numpy()
        total += frames.shape[0]
        pbar.update(1)
    pbar.close()
    return 100.*correct/total


def train(model=None, dataloader=None, optim=None,
          scheduler=None, criterion=nn.CrossEntropyLoss(), losses=[],
          averages=[], n_epochs=0, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    meter = AverageMeter()
    if verbose is False:
        pbar = tqdm(total=len(dataloader))
    for epoch in range(n_epochs):
        for i_batch, batch in enumerate(dataloader):
            frames, lbls = batch
            frames, lbls = frames.to(device), lbls.to(device)
            model.train()
            optim.zero_grad()
            model.lstm.reset_hidden_state()
            # Establish shared key
            predictions = model(frames)
            # print(predictions.shape)
            #print(predictions, lbls)
            loss = criterion(predictions, lbls)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            meter.update(loss.item(), frames.shape[0])
            if verbose:
                print(
                    '[{}/{}] Message loss:{:.4f} '.format(i_batch,
                                                          len(dataloader),
                                                          loss.item()))
            else:
                pbar.update(1)
        print('{} Average:{:.4f} '.format(epoch, meter.avg))
        if verbose is False:
            pbar.reset(0)
        if scheduler is not None:
            scheduler.step(meter.avg)
            averages.append(meter.avg)
            meter.reset()
