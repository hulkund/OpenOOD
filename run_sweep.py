from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from torchvision.transforms import RandAugment
from torchvision import transforms
from openood.postprocessors import get_postprocessor
import pandas
import os
import numpy as np


config_files = [
        './configs/datasets/cifar100/cifar100.yml',
        './configs/datasets/cifar100/cifar100_ood.yml',
        './configs/networks/resnet18_32x32.yml',
        './configs/pipelines/test/test_ood.yml',
        './configs/preprocessors/base_preprocessor.yml',
        './configs/postprocessors/mls.yml',
        './configs/rand_augment/base.yml',
]

def main(config_files):
    # config.parse_refs()

    # load config files for cifar100 baseline
    PATH="/home/gridsan/nhulkund/OpenOOD/"
    config = config.Config(*config_files)
    # modify config 
    #config.network.checkpoint = PATH+'/results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
    if config.dataset.name == 'cifar100':
        config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/cifar100_res18_acc78.20.ckpt'
    elif config.dataset.name == 'cifar10':
        config.network.checkpoint = PATH+'/results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
    elif config.dataset.name == 'imagenet':
        config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/imagenet_res50_acc76.10.pth'
    config.network.pretrained = True
    config.num_workers = 8
    config.save_output = True
    config.parse_refs()

    acc_metrics, ood_metrics = get_metrics()
    

def get_metrics():
    # get dataloader
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    # init network
    net = get_network(config.network).cuda()
    # init ood evaluator
    evaluator = get_evaluator(config)
    postprocessor = get_postprocessor(config)
    acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'], postprocessor)
    ood_metrics = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
    return acc_metrics, ood_metrics


# def save_metrics(config, ood_metrics):
#     save_dir = os.path.join(config.output_dir, 'scores')
#     os.makedirs(save_dir, exist_ok=True)
#     df = pd.DataFrame(ood_metrics)



