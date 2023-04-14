from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class MaxLogitPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, augmentation=False):
        output = net(data)
        if self.config.rand_augment.augmentation == True and augmentation:
            #print("initial output size", output.shape)
            output=output.cpu()
            num_imgs = self.config.rand_augment.num_augments+1
            if output.shape[0] % num_imgs == 0: 
                first_dim = int(output.shape[0]/num_imgs)
                second_dim = num_imgs
            else:
                raise TypeError
            if self.config.rand_augment.averaging_before_max == True:
                output_reshaped = torch.reshape(output, (first_dim, second_dim, output.shape[1]))
                #print("output after reshaping", output_reshaped.shape)
                output_averaged = torch.Tensor(torch.mean(output_reshaped,dim=1))
                #print("output after averaging", output_averaged.shape)
                conf, pred = torch.max(output_averaged, dim=1)
                #print("output maxed", pred.shape)
            else:
                conf, pred = torch.max(output, dim=1)
                pred_reshaped = pred.numpy().reshape(first_dim,second_dim)
                pred_averaged = torch.Tensor(np.mean(pred_reshaped,axis=1))
                conf_reshaped = conf.numpy().reshape(first_dim,second_dim)
                conf_averaged = torch.Tensor(np.mean(conf_reshaped,axis=1))
                conf, pred = conf_averaged, pred_averaged
        else:
            conf, pred = torch.max(output, dim=1)
        return pred, conf
