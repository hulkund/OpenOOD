import subprocess
import shlex
import torch
from os.path import exists


def perform_sweep():
    for ood_metric in ['msp']:
        for dataset in ['cifar10']:#['cifar10', 'cifar100']:#['imagenet']:
            for rand_augment in ['avg_min','min_avg','base','max']:
                for rand_n in [1]:
                    for rand_m in  [5,10,15]:
                        print(ood_metric, dataset, rand_augment, rand_n, rand_m)
                        subprocess.call(shlex.split('sbatch get_ood_metrics.sh "%s" "%s" "%s" %s %s'%(ood_metric, dataset, rand_augment, rand_n, rand_m)))

if __name__ == "__main__":
    perform_sweep()

# def config_sweep():
#     id_datasets = ['cifar10', 'cifar100', 'imagenet']
#     postprocessors = ['dice','ebo','gradnorm','mds','mls','msp','odin','react','gram']
#     rand_augments = ['base', 'base_max_before_average', 'no_augmentation']

# still left: mds, odin
