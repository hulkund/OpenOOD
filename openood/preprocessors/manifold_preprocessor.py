import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict
from torchvision.transforms import RandAugment
import torch
from torch import Tensor


class ManifoldPreprocessor():
    """For train dataset standard transformation."""
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        normalization_type = config.dataset.normalization_type
        self.rand_n = config.preprocessor.preprocessor_args.rand_n
        self.rand_m = config.preprocessor.preprocessor_args.rand_m
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(self.image_size, padding=4),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
            RandAugmentManifold(rand_n=self.rand_n,rand_m=self.rand_m),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)
    

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

