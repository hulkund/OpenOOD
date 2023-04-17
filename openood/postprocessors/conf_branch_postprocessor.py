from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .randaugment_manifold_function import RandAugmentManifold, postprocess_augmentations



class ConfBranchPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ConfBranchPostprocessor, self).__init__(config)
        self.config = config

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, conf = net(data, return_confidence=True)
        conf = torch.sigmoid(conf)
        if self.config.rand_augment.augmentation and augmentation:
            _, pred = postprocess_augmentations(self.config, score)
        else:
            _, pred = torch.max(score, dim=1)
        #_, pred = torch.max(output, dim=1)
        return pred, conf
