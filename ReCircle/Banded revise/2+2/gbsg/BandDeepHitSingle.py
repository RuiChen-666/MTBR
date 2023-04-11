# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:40:36 2022

@author: 89696
"""

import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from pycox import models
from Loss import DeepHitSingleLoss
#from pycox.models.utils import pad_col
from pycox.models.utils import pad_col,cumsum_reverse

class BandedDeepHitSingle(models.pmf.PMFBase):
    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None):
        if loss is None:
            #loss = models.loss.DeepHitSingleLoss(alpha, sigma)
            loss = DeepHitSingleLoss(alpha,sigma)
        super().__init__(net, loss, optimizer, device, duration_index)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
    
    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        
        pre_preds = pad_col(preds,val=0.)[:,:-1]
        post_preds = pad_col(preds,val=1.)[:,1:]
        
        '''
        这里是需要修改的位置
        请确认 preds前的系数 = pre_preds前的系数 + post_preds前的系数
        '''
        avg_preds = (2*pre_preds + 2*post_preds + 8*preds) / 12
        end_preds = cumsum_reverse(avg_preds, dim=1)
        
        pmf = pad_col(end_preds).softmax(1)[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)
 