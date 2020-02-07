from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .utils import save_checkpoint, plot_losses

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
    model.to(device)
    model.eval()
    correct = 0
    loss = 0
    total = 0
    print('------------ Validating performance')
    for idx, batch in enumerate(dataloader):
        frames, lbls = batch
        frames, lbls = frames.to(device), lbls.float().to(device)
        with torch.no_grad():
            model.lstm.reset_hidden_state()
            predictions = model(frames)
            correct += (torch.round(predictions).detach()
                        == lbls).sum().cpu().numpy()
        total += frames.shape[0]
        loss += (lbls.shape[0]) * \
            (criterion(predictions, lbls).detach().cpu().item())
    return 100.*correct/total, loss/total


def train_dfd(model=None, dataloader=None, testloader=None, optim=None,
              scheduler=None, criterion=nn.CrossEntropyLoss(), losses=[],
              averages=[], n_epochs=0, e_saves=1, device='cpu', verbose=False):
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
    best_loss = 999
    now = datetime.now()
    chpt_dest = './checkpoints_{}'.format(now.strftime("%d-%m-%Y_%H:%M:%S"))
    Path(chpt_dest).mkdir(parents=True, exist_ok=True)
    test_losses = []
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
                               criterion=criterion, device=device)
        test_losses.append(v_loss)
        if (epoch+1) % e_saves == 0:
            save_checkpoint(model, 'Epoch {} Train:{} Test:{} '.format(epoch, meter.avg, v_loss),
                            '{}/model_epoch_{}.pth.tar'.format(chpt_dest, epoch+1))
        if v_loss < best_loss:
            best_loss = v_loss
            save_checkpoint(model, 'best model Train:{} Test: {}'.format(meter.avg, v_loss),
                            '{}/best_model.pth.tar'.format(chpt_dest))
        print('Epoch {} \n Train loss:{:.4f} \n Test accuracy:{:.4f} loss:{:.4f}'.format(
            epoch, meter.avg, acc, v_loss))
        if verbose is False:
            pbar.refresh()
            pbar.reset()
        if scheduler is not None:
            scheduler.step(meter.avg)
            averages.append(meter.avg)
            meter.reset()
    pbar.close()
    plot_losses(averages, test_losses, chpt_dest)
    save_checkpoint(
        model=model,
        description='Final Train:{} Test:{}'.format(
            averages[-1], test_losses[-1]),
        filename='{}/final_model.pth.tar'.format(chpt_dest))
