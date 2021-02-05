'''
Adapted from 
https://github.com/ksnzh/DORN.pytorch/blob/master/ordinal_regression_loss.py
'''

import torch
from torch import nn
from torch.autograd import Variable

IGNORE = -100

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, K):
        super(OrdinalRegressionLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE)
        self.K = K

    def forward(self, pred, target):
        """
        :param pred: N * k
        :param target: N
        :return:
        """
        raise NotImplementedError
        '''
        loss = 0
        for k in range(self.K):
            
            # Compute loss for target < k
            mask_lt = torch.gt(target, k)       # if target > k, no loss
            target_gt = torch.zeros_like(target)
            target_gt[mask_lt.data] = IGNORE    # ignore all target > k samples

            loss += self.cross_entropy(feature.narrow(1, 2*k, 2), target_lt) / (count_lt + 1)


            mask_lt = torch.gt(target, k)   # k-th: the current k is less than the gt label
            mask_gt = torch.le(target, k)   #

            # Generate "ground truth"
            # (1) truth for pred < target, ignore target
            target_lt = torch.ones_like(target)
            target_lt[mask_gt.data] = IGNORE

            target_gt = torch.zeros_like(target)
            target_gt[mask_lt.data] = IGNORE

            mask_lt = mask_lt.type(torch.cuda.FloatTensor).unsqueeze(1)
            count_lt = torch.sum(mask_lt)
            count_gt = mask_lt.size(0) * mask_lt.size(2) * mask_lt.size(3) - count_lt

            # feature_k = feature.narrow(1, 2*k, 2)

            loss += self.cross_entropy(feature.narrow(1, 2*k, 2), target_lt) / (count_lt + 1)
            loss += self.cross_entropy(feature.narrow(1, 2*k, 2), target_gt) / (count_gt + 1)

        return loss

        '''