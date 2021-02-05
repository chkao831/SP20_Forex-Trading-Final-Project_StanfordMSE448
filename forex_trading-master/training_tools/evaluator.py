import argparse
import os
import random
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from sklearn.metrics import roc_auc_score, accuracy_score

from .meters import AverageMeter, ProgressMeter
from .custom_loss_functions import *


LOSS_MAP = {'mse': nn.MSELoss(),
            'ordinal': OrdinalRegressionLoss(7),
            'ce': nn.CrossEntropyLoss()
            }


def metric_mse(output, target, *args):
    l = nn.functional.mse_loss(output, target).detach().cpu().numpy()
    return l


def metric_accuracy(output, target, *args):
    output = np.argmax(output.detach().cpu().numpy(), axis=1)
    target = target.detach().cpu().numpy().astype('int')

    return accuracy_score(output, target)



class Evaluator:

    def __init__(self, args, model, loaders, metrics):

        self.model = model
        
        self.loss_name = args.train_args.loss_func
        self.loss_func = self._get_loss(args.train_args.loss_func)
        self.loaders = loaders

        self.args = args
        self.logger = args.misc_args.logger

        self.metrics = metrics
        self.metric_best_vals = {metric: 0 for metric in self.metrics}

    def _get_loss(self, loss_func):
        
        if loss_func in LOSS_MAP:
            return LOSS_MAP[loss_func]
        else:
            raise NotImplementedError(f'Loss {loss_func} is not supported yet')

    def evaluate(self, eval_type, epoch, freeze_weights):

        self.logger.log_stdout(f'Evaluation for {eval_type}, epoch {epoch}')

        loader = self.loaders[eval_type]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter(f'Loss ({self.loss_name})', ':.4e')

        metric_meters = {metric: AverageMeter(metric, self.metrics[metric]['format']) \
                                                    for metric in self.metrics}
        list_meters = [metric_meters[m] for m in metric_meters]

        progress = ProgressMeter(self.logger,
                                 len(loader),
                                 [batch_time, losses, *list_meters],
                                 prefix=f'{eval_type}@Epoch {epoch}: ')

        if freeze_weights:
            # switch to evaluate mode
            self.model.eval()

        with torch.no_grad():
            end = time.time()
            i = 0
            for eval_x, eval_gt in loader:
                #if self.args.gpu is not None:
                #    images = images.cuda(self.args.gpu, non_blocking=True)
                #eval_gt = eval_gt.cuda(self.args.gpu, non_blocking=True)

                eval_x = eval_x.to(self.args.train_args.device).float()
                # eval_gt = eval_gt.to(self.args.train_args.device).float()
                eval_gt = eval_gt.to(self.args.train_args.device).long()

                # compute output
                pred = self.model(eval_x)
                loss = self.loss_func(pred, eval_gt)

                # JBY: For simplicity do losses first
                losses.update(loss.item(), eval_x.size(0))

                for metric in self.metrics:
                    args = [pred, eval_gt, *self.metrics[metric]['args']]
                        
                    metric_func = globals()[self.metrics[metric]['func']]
                    result = metric_func(*args)
                    
                    metric_meters[metric].update(result, eval_x.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.train_args.disp_steps == 0:
                    progress.display(i)
                i += 1
        progress.display(i)

        for metric in self.metrics:
            self.metric_best_vals[metric] = max(metric_meters[metric].avg,
                                                self.metric_best_vals[metric])
    