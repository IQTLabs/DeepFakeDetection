from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ['train_dfd', 'test_dfd']


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


def test_dfd(dataloader, model, criterion, device):
    """ Test deep fake detector
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader used for evaluation
    model : torch.Module
        Pytorch module to evaluate
    crriterion : torch.nn.Module
        Objective function used in training
    device : str
        Device to run eval
    Returns
    -------
    accuracy : float
        Model accuracy over evaluation set
    """
    device = torch.device(device)
    pbar = tqdm(total=len(dataloader))
    model.to(device)
    model.eval()
    correct = 0
    loss = 0
    total = 0
    for idx, batch in enumerate(dataloader):
        frames, lbls = batch
        frames, lbls = frames.to(device), lbls.float().to(device)
        with torch.no_grad():
            model.lstm.reset_hidden_state()
            predictions = model(frames)
            correct = (torch.round(predictions).detach()
                       == lbls).sum().cpu().numpy()
        total += frames.shape[0]
        loss += (lbls.shape[0]) * \
            (criterion(predictions, lbls).detach().cpu().item())
        pbar.update(1)
    pbar.close()
    return 100.*correct/total, loss/total


def train_dfd(model=None, dataloader=None, testloader=None, optim=None,
              scheduler=None, criterion=nn.CrossEntropyLoss(), losses=[],
              averages=[], n_epochs=0, device='cpu', verbose=False):
    """ Training routing for deep fake detector
    Parameters
    ----------
    model : torch.Module
        Deep fake detector model
    dataloader : torch.utils.data.DataLoader
        Training dataset
    optim : torch.optim
        Optimizer for pytorch model
    scheduler : torch.optim.lr_scheduler
        Optional learning rate scheduler for the optimizer
    criterion : torch.nn.Module
        Objective function for optimization
    losses : list
        List to hold the lossses over each mini-batch
    averages : list
        List to hold the average loss over each epoch
    n_epochs : int
        Number of epochs for training
    device : str
        Device to run training procedure
    verbose : bool
        Verbose switch to print losses at each mini-batch
    """
    device = torch.device(device)
    model = model.to(device)
    meter = AverageMeter()
    if verbose is False:
        pbar = tqdm(total=len(dataloader))
    for epoch in range(n_epochs):
        for i_batch, batch in enumerate(dataloader):
            frames, lbls = batch
            frames, lbls = frames.to(device), lbls.float().to(device)
            model.train()
            optim.zero_grad()
            model.lstm.reset_hidden_state()
            # Establish shared key
            predictions = model(frames)
            # print(predictions.shape)
            # print(predictions, lbls)
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
        acc, v_loss = test_dfd(dataloader=testloader, model=model,
                               criterion=criterion, device='cuda:1')
        print('{} Train average:{:.4f} \n Test average{:.4f} average {:.4f}'.format(
            epoch, meter.avg, acc, v_loss))
        if verbose is False:
            pbar.reset(0)
        if scheduler is not None:
            scheduler.step(meter.avg)
            averages.append(meter.avg)
            meter.reset()
    pbar.close()
