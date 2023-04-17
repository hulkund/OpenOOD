
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from torchvision.transforms import RandAugment
from torchvision import transforms
from openood.postprocessors import get_postprocessor
import pandas as pd
import os
import numpy as np
import argparse
from openood.utils import config as cfg
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--ood_metric')
parser.add_argument('--rand_augment')
parser.add_argument('--rand_n')
parser.add_argument('--rand_m')
args = parser.parse_args()


PATH="/home/gridsan/nhulkund/OpenOOD/"
# writer = SummaryWriter()

def fill_config_files(dataset, ood_metric, rand_augment):
    networks = {'cifar100':'./configs/networks/resnet18_32x32.yml', 'cifar10':'./configs/networks/resnet18_32x32.yml', 'imagenet':'/scripts/download/results/checkpoints/imagenet_res50_acc76.10.pth'}
    config_files = [
        './configs/datasets/{}/{}.yml'.format(dataset, dataset),
        './configs/datasets/{}/{}_ood.yml'.format(dataset, dataset),
        networks[dataset],
        './configs/pipelines/test/test_ood.yml',
        './configs/preprocessors/base_preprocessor.yml',
        './configs/postprocessors/{}.yml'.format(ood_metric),
        './configs/rand_augment/{}.yml'.format(rand_augment),
    ]
    return config_files

def get_config():
    config_files = fill_config_files(args.dataset, args.ood_metric, args.rand_augment)
    config = cfg.Config(*config_files)
    config.network.pretrained = True
    config.num_workers = 8
    config.save_output = False
    config.parse_refs()
    config.rand_augment.rand_n = int(args.rand_n)
    config.rand_augment.rand_m = int(args.rand_m)
    if args.dataset == 'cifar100':
        config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/cifar100_res18_acc78.20.ckpt'
    elif args.dataset == 'cifar10':
        config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/cifar10_res18_acc94.30.ckpt'
    elif args.dataset == 'imagenet':
        config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/imagenet_res50_acc76.10.pth'
    return config
    
def run_metrics(config):
    # get dataloader
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    print("got data")
    # init network
    net = get_network(config.network).cuda()
    print("got network")
    # init ood evaluator
    evaluator = get_evaluator(config)
    postprocessor = get_postprocessor(config)
    print("got evaluator, postprocessor")
    acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'], postprocessor)
    print("acc metric")
    ood_metrics = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
    print("ood metric")
    return acc_metrics, ood_metrics

if __name__ == "__main__":
    config_exp = get_config()
    print("got config")
    acc_metrics, ood_metrics = run_metrics(config_exp)
    exp_name = '{}_{}_m={}_n={}_{}'.format(config_exp.dataset.name, config_exp.postprocessor.name, config_exp.rand_augment.rand_n, config_exp.rand_augment.rand_m, config_exp.rand_augment.name)
    print("ran metrics")
    print(args.dataset, args.ood_metric, args.rand_augment, "rand_n=", args.rand_n, " rand_m=", args.rand_m)
    # writer.add_scalar('OOD metric', ood_metrics[1], exp_name)
    ood_df = pd.DataFrame(ood_metrics, index=[0])
    acc_df = pd.DataFrame(acc_metrics, index=[0])
    filename = PATH+'results/hparam/'+exp_name
    ood_df.to_csv(filename)

# def save_metrics(config, ood_metrics):
#     save_dir = os.path.join(config.output_dir, 'scores')
#     os.makedirs(save_dir, exist_ok=True)
#     df = pd.DataFrame(ood_metrics)



