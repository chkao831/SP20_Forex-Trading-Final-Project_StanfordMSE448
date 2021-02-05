import sys
sys.path.append('..')

import os

import pandas as pd
import tqdm

from arguments import *
from constants import *
from data import *
import models as supported_models
from training_tools import *
from utils import *


storage = GCStorage.get_CloudFS(project_name=PROJECT_NAME,
                                bucket_name=GC_BUCKET,
                                credential_path=CREDENTIAL_PATH)

all_args = parse_args()
# all_args = parse_from_string('--exp_name=jupyter --device=cpu --num_candles=20 --num_workers=1 --fast_debug=True --candle_interval=30 --log_level=4')

logger = all_args.misc_args.logger
device = all_args.train_args.device

train_loader, valid_loader = get_dataloaders(all_args)
loaders = {'train': train_loader, 'valid': valid_loader}

model_init_func = supported_models.__dict__[all_args.model_args.model_type]
model = model_init_func(all_args.model_args) # TODO: JBY. Enable Dataparallel, if needed.
model = model.to(device)

logger.log_stdout(model)

optimizer = Optimizer(all_args, model.parameters(), len(train_loader.dataset))
evaluator = Evaluator(all_args, model, loaders, EVAL_METRICS)

# Run eval prior to training
evaluator.evaluate('valid', 0, freeze_weights=True)

##### Train #####
for epoch in range(all_args.train_args.max_epochs):

    logger.log_stdout(f'=== Epoch [{epoch}] === ')

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter(f'Loss ({all_args.train_args.loss_func})', ':.4e')

    metric_meters = {metric: AverageMeter(metric, evaluator.metrics[metric]['format']) \
                                                for metric in evaluator.metrics}
    list_meters = [metric_meters[m] for m in metric_meters]

    progress = ProgressMeter(logger,
                             len(train_loader),
                             [batch_time, losses, *list_meters],
                             prefix=f'train@Epoch {epoch}: ')

    end = time.time()
    i = 0
    for train_x, train_gt in train_loader:
        model.train()
        #if all_args.train_args.device is not None:
        #    train_x = train_x.cuda(all_args.gpu, non_blocking=True)
        #train_gt = train_gt.cuda(all_args.gpu, non_blocking=True)

        train_x = train_x.to(device).float()
        # train_gt = train_gt.to(device).float()
        train_gt = train_gt.to(device).long()

        optimizer.zero_grad()

        # compute output
        pred = model(train_x)
        loss = evaluator.loss_func(pred, train_gt)

        # JBY: For simplicity do losses first
        losses.update(loss.item(), train_x.size(0))

        for metric in EVAL_METRICS:
            args = [pred, train_gt, *evaluator.metrics[metric]['args']]
                
            metric_func = globals()[evaluator.metrics[metric]['func']]
            result = metric_func(*args)
            
            metric_meters[metric].update(result, train_x.size(0))

        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % all_args.train_args.disp_steps == 0:
            progress.display(i)

        if i % all_args.train_args.eval_steps == 0 and i != 0:
            evaluator.evaluate('valid', epoch, freeze_weights=True)
        
        i += 1
    progress.display(i)
    
    evaluator.evaluate('valid', epoch, freeze_weights=True)    # certainly evaluate ever epoch