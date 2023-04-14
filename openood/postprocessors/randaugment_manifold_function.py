from torchvision.transforms import RandAugment
import torch
from torch import Tensor
import numpy as np

class RandAugmentManifold(object):
    def __init__(self, rand_n, rand_m):
        self.n = rand_n
        self.m = rand_m

    def preprocess(self, data, data_max, data_min):
        data = (data - data_min) / (data_max - data_min) # normalize the data to 0 - 1
        data = 255 * data # Now scale by 255
        data = data.type(torch.uint8)
        return data

    def postprocess(self, data, data_max, data_min):
        data = data.type(torch.float32)
        data /= 255
        data = (data * (data_max - data_min)) + data_min
        return data

    def augment_preprocess(self, im):
        rand_aug = RandAugment(num_ops=self.n, magnitude=self.m)
        data_max, data_min = im.clone().max(), im.clone().min()
        pre_im = self.preprocess(im, data_max, data_min)
        aug_im = rand_aug(pre_im)
        post_im = self.postprocess(aug_im, data_max, data_min)
        return post_im
    
    def __call__(self, img: Tensor) -> Tensor:
        img = self.augment_preprocess(img)
        return img 

def postprocess_augmentations(config, output):
    #print("initial output size", output.shape)
    output=output.cpu()
    num_imgs = config.rand_augment.num_augments+1
    if output.shape[0] % num_imgs == 0: 
        first_dim = int(output.shape[0]/num_imgs)
        second_dim = num_imgs
    else:
        raise TypeError
    if config.rand_augment.averaging_before_max == True:
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
    return conf, pred