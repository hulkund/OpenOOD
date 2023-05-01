"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .randaugment_manifold_function import RandAugmentManifold, postprocess_augmentations

class ODINPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.temperature = self.args.temperature
        self.noise = self.args.noise
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any, augmentation):
        if self.config.rand_augment.per_gradient_odin == True and augmentation:
            data_for_grad = []
            for i in range(0,data.shape[0],11):
                data_for_grad.append(data[i].cpu())
            data_for_grad=torch.stack(data_for_grad).cuda()

            data_for_grad.requires_grad = True
            data.requires_grad = True

            output = net(data)
            output_for_grad = net(data_for_grad)

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            criterion = nn.CrossEntropyLoss()

            labels = output_for_grad.detach().argmax(axis=1)

            # Using temperature scaling
            output = output / self.temperature
            output_for_grad = output_for_grad / self.temperature

            loss = criterion(output_for_grad, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(data_for_grad.grad.detach(), 0)
            gradient = (gradient.float() - 0.5) * 2

            # Scaling values taken from original code
            gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
            gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
            gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

            stacked_gradient = []
            for i in range(0,len(gradient)):
                stacked_gradient.append(gradient[i])
                for j in range(self.config.rand_augment.num_augments):
                    stacked_gradient.append(gradient[i])
            gradient = torch.stack(stacked_gradient)

            # Adding small perturbations to images
            tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
            output = net(tempInputs)
            output = output / self.temperature

            # Calculating the confidence after adding perturbations
            nnOutput = output.detach()
            nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
            nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
            if self.config.rand_augment.augmentation:
                conf, pred = postprocess_augmentations(self.config, nnOutput)
            else:
                conf, pred = nnOutput.max(dim=1)

        else:
            data.requires_grad = True

            output = net(data)
            #output_for_grad = net(data_for_gradient)

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            criterion = nn.CrossEntropyLoss()

            labels = output.detach().argmax(axis=1)

            # Using temperature scaling
            output = output / self.temperature

            loss = criterion(output, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(data.grad.detach(), 0)
            gradient = (gradient.float() - 0.5) * 2

            # Scaling values taken from original code
            gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
            gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
            gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

            # Adding small perturbations to images
            tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
            output = net(tempInputs)
            output = output / self.temperature

            # Calculating the confidence after adding perturbations
            nnOutput = output.detach()
            nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
            nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

            if self.config.rand_augment.augmentation and augmentation:
                conf, pred = postprocess_augmentations(self.config, nnOutput)
            else:
                conf, pred = nnOutput.max(dim=1)

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]
