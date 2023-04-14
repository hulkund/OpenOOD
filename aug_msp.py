from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from torchvision.transforms import RandAugment
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from PIL import Image
from openood.evaluators.metrics import compute_all_metrics

# # load config files for cifar10 baseline
PATH="/home/gridsan/nhulkund/OpenOOD/"
# config_files = [
#     './configs/datasets/cifar10/cifar10.yml',
#     './configs/datasets/cifar10/cifar10_ood.yml',
#     './configs/networks/resnet18_32x32.yml',
#     './configs/pipelines/test/test_ood.yml',
#     './configs/preprocessors/base_preprocessor.yml',
#     './configs/postprocessors/msp.yml',
# ]
# config = config.Config(*config_files)
# # modify config 
# #config.network.checkpoint = PATH+'/results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
# config.network.checkpoint = PATH+'/scripts/download/results/checkpoints/cifar10_res18_acc94.30.ckpt'
# config.network.pretrained = True
# config.num_workers = 8
# config.save_output = False
# config.parse_refs()

def main(config):
    # get dataloader
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    # init network
    net = get_network(config.network).cuda()
    # init ood evaluator
    evaluator = get_evaluator(config)
    # save root
    save_root = PATH+f'/results/{config.exp_name}'
    # in distribution
    in_distribution(net, id_loader_dict, save_root)
    # ood detection
    ood_detection(net, ood_loader_dict, save_root)
    # get results
    results = load_results(save_root)
    postprocess_results(save_root)
    eval_ood(postprocess_results)

def augment_preprocess(im):
    rand_aug = RandAugment(num_ops=1, magnitude=5)
    data_max, data_min = im.clone().max(), im.clone().min()
    pre_im = preprocess(im, data_max, data_min)
    aug_im = rand_aug(pre_im)
    post_im = postprocess(aug_im, data_max, data_min)
    return post_im

def preprocess(data, data_max, data_min):
    data = (data - data_min) / (data_max - data_min) # normalize the data to 0 - 1
    data = 255 * data # Now scale by 255
    data = data.type(torch.uint8)
    return data

def postprocess(data, data_max, data_min):
    data = data.type(torch.float32)
    data /= 255
    data = (data * (data_max - data_min)) + data_min
    return data

def save_arr_to_dir(arr, dir):
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir, 'wb+') as f:
        np.save(f, arr)

def in_distribution(net, id_loader_dict, save_root):
    # save id (test & val) results
    net.eval()
    modes = ['test', 'val']
    for mode in modes:
        dl = id_loader_dict[mode]
        dataiter = iter(dl)
        
        logits_list = []
        feature_list = []
        label_list = []
        id_list = []
        print(id_list)
        for i in tqdm(range(1,
                        len(dataiter) + 1),
                        desc='Extracting reults...',
                        position=0,
                        leave=True):
            batch = next(dataiter)
            data = batch['data'].cuda()
            label = batch['label']
            with torch.no_grad():
                logits_cls, feature = net(data, return_feature=True)
            logits_list.append(logits_cls.data.to('cpu').numpy())
            feature_list.append(feature.data.to('cpu').numpy())
            label_list.append(label.numpy())

        logits_arr = np.concatenate(logits_list)
        feature_arr = np.concatenate(feature_list)
        label_arr = np.concatenate(label_list)
        
        save_arr_to_dir(logits_arr, osp.join(save_root, 'id', f'{mode}_logits.npy'))
        save_arr_to_dir(feature_arr, osp.join(save_root, 'id', f'{mode}_feature.npy'))
        save_arr_to_dir(label_arr, osp.join(save_root, 'id', f'{mode}_labels.npy'))

def ood_detection(net, ood_loader_dict, save_root):
    # save ood results
    net.eval()
    ood_splits = ['nearood', 'farood']
    for ood_split in ood_splits:
        for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
            dataiter = iter(ood_dl)
        
            logits_list = []
            feature_list = []
            label_list = []
            id_list = []

            for i in tqdm(range(1,
                            len(dataiter) + 1),
                            desc='Extracting reults...',
                            position=0,
                            leave=True):
                batch = next(dataiter)
                data = batch['data'].cuda()
                label = batch['label']
                aug_data=[]
                aug_label=[]
                for k in range(data.shape[0]):
                    count+=1
                    orig_img=data[k]
                    aug_data.append(orig_img.cpu())
                    aug_label.append(label[k])
                    for j in range(10):
                        id_list.append(count)
                        aug_img=augment_preprocess(orig_img.cpu())
                        aug_img=torch.Tensor(aug_img)
                        aug_data.append(aug_img)
                        aug_label.append(label[k])
                aug_data=torch.stack(aug_data).cuda()
                aug_label=torch.stack(aug_label)
                with torch.no_grad():
                    logits_cls, feature = net(aug_data, return_feature=True)
                logits_list.append(logits_cls.data.to('cpu').numpy())
                feature_list.append(feature.data.to('cpu').numpy())
                #label_list.append(aug_label.numpy())
                label_list.append(label.numpy())

            logits_arr = np.concatenate(logits_list)
            feature_arr = np.concatenate(feature_list)
            label_arr = np.concatenate(label_list)
            
            save_arr_to_dir(logits_arr, osp.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
            save_arr_to_dir(feature_arr, osp.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
            save_arr_to_dir(label_arr, osp.join(save_root, ood_split, f'{dataset_name}_labels.npy'))
            save_arr_to_dir(id_list, osp.join(save_root, ood_split, f'{dataset_name}_ids.npy'))

# build msp method (pass in pre-saved logits)
def msp_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    conf, pred = torch.max(score, dim=1)
    return pred, conf

def msp_aug_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    if score.shape[0] % 11 == 0: 
        first_dim=int(score.shape[0]/11)
        second_dim=11
    else:
        print(score.shape)
    score_reshaped=score.numpy().reshape(first_dim,second_dim,10)
    score_averaged=torch.Tensor(np.mean(score_reshaped,axis=1))
    conf, pred = torch.max(score_averaged, dim=1)
    return pred, conf

def load_results(save_root):
    # load logits, feature, label for this benchmark
    results = dict()
    # for id
    modes = ['val', 'test']
    results['id'] = dict()
    for mode in modes:
        results['id'][mode] = dict()
        results['id'][mode]['feature'] = np.load(osp.join(save_root, 'id', f'{mode}_feature.npy'))
        results['id'][mode]['logits'] = np.load(osp.join(save_root, 'id', f'{mode}_logits.npy'))
        results['id'][mode]['labels'] = np.load(osp.join(save_root, 'id', f'{mode}_labels.npy'))

    # for ood
    split_types = ['nearood', 'farood']
    for split_type in split_types:
        results[split_type] = dict()
        dataset_names = config['ood_dataset'][split_type].datasets
        for dataset_name in dataset_names:
            results[split_type][dataset_name] = dict()
            results[split_type][dataset_name]['feature'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_feature.npy'))
            results[split_type][dataset_name]['logits'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_logits.npy'))
            results[split_type][dataset_name]['labels'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_labels.npy'))
    return results



def postprocess_results_msp(results):
    # get pred, conf, gt from MSP postprocessor (can change to your custom_postprocessor here)
    postprocess_results = dict()
    # id
    modes = ['val', 'test']
    postprocess_results['id'] = dict()
    for mode in modes:
        pred, conf = msp_postprocess(torch.from_numpy(results['id'][mode]['logits']))
        pred, conf = pred.numpy(), conf.numpy()
        gt = results['id'][mode]['labels']
        postprocess_results['id'][mode] = [pred, conf, gt]

    # ood
    split_types = ['nearood', 'farood']
    for split_type in split_types:
        postprocess_results[split_type] = dict()
        dataset_names = config['ood_dataset'][split_type].datasets
        for dataset_name in dataset_names:
            pred, conf = msp_aug_postprocess(torch.from_numpy(results[split_type][dataset_name]['logits']))
            pred, conf = pred.numpy(), conf.numpy()
            gt = results[split_type][dataset_name]['labels']
            gt = -1 * np.ones_like(gt)   # hard set to -1 here
            postprocess_results[split_type][dataset_name] = [pred, conf, gt]
    return postprocess_results

def print_nested_dict(dict_obj, indent = 0):
    ''' Pretty Print nested dictionary with given indent level  
    '''
    # Iterate over all key-value pairs of dictionary
    for key, value in dict_obj.items():
        # If value is dict type, then print nested dict 
        if isinstance(value, dict):
            print(' ' * indent, key, ':', '{')
            print_nested_dict(value, indent + 2)
            print(' ' * indent, '}')
        else:
            print(' ' * indent, key, ':', value.shape)

def eval_ood(postprocess_results):
    [id_pred, id_conf, id_gt] = postprocess_results['id']['test']
    split_types = ['nearood', 'farood']

    for split_type in split_types:
        metrics_list = []
        print(f"Performing evaluation on {split_type} datasets...")
        dataset_names = config['ood_dataset'][split_type].datasets
        
        for dataset_name in dataset_names:
            [ood_pred, ood_conf, ood_gt] = postprocess_results[split_type][dataset_name]

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            print_all_metrics(ood_metrics)
            metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)   
        print_all_metrics(metrics_mean)


            

        