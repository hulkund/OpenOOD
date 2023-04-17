from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .randaugment_manifold_function import RandAugmentManifold, postprocess_augmentations

class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, augmentation=False):
        output = net(data)
        score = torch.softmax(output, dim=1)
        if self.config.rand_augment.augmentation and augmentation:
            conf, pred = postprocess_augmentations(self.config, score)
        else:
            conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader, augmentation=False):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = batch['data'].cuda()
            label = batch['label'].cuda()

            if self.config.rand_augment.augmentation == True and augmentation:
                #print("rand aug")
                aug_data = []
                aug_label = []
                #print("augmenting")
                rand_aug=RandAugmentManifold(self.config.rand_augment.rand_n, self.config.rand_augment.rand_m)
                for k in range(data.shape[0]):
                    orig_img=data[k]
                    aug_data.append(orig_img.cpu())
                    for j in range(self.config.rand_augment.num_augments):
                        aug_img=rand_aug(orig_img.cpu())
                        aug_img=torch.Tensor(aug_img)
                        aug_data.append(aug_img)
                        aug_label.append(label[k])
                aug_data=torch.stack(aug_data).cuda()
                #print("done augmenting")
                data = aug_data
                #label = aug_label

            pred, conf = self.postprocess(net, data, augmentation)

            if self.config.rand_augment.augmentation == True and augmentation:
                len_data = int(len(data)/(self.config.rand_augment.num_augments+1))
            else:
                len_data = len(data)
            
            for idx in range(len_data):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
