import time

import torchvision.transforms as standard_transforms
from PIL import Image, ImageEnhance
import random

from dataset.CityscapesLoader_old import *


class Compose(object):
    def __init__(self, transforms, debug=False):
        self.transforms = transforms
        self.debug=debug

    def __call__(self, img, mask):
        if mask is not None:
            assert img.size == mask.size
        total_time1 = time.process_time()
        for t in self.transforms:
            t1 = time.process_time()
            img, mask = t(img, mask, self.debug)
            t2=time.process_time()
            if self.debug:
                print(t, " : ",t2-t1)

        total_time2 = time.process_time()
        if self.debug:
            print("Total time: ",total_time2-total_time1)
        return img, mask

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, lbl, debug):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        if debug:
            print("Rotate: ",rotate_degree)

        img_rotate = img.rotate(rotate_degree, Image.BILINEAR)

        lbl_rotate = Image.eval(lbl, lambda px: 254 if px == 0 else px)
        lbl_rotate = lbl_rotate.rotate(rotate_degree, Image.NEAREST)
        lbl_rotate = Image.eval(lbl_rotate, lambda px: 255 if px == 0 else px)
        lbl_rotate = Image.eval(lbl_rotate, lambda px: 0 if px == 254 else px)
        return img_rotate, lbl_rotate


class RandomHorizontallyFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img, lbl, debug):
        if random.random() < self.prob:
            if debug:
                print("Flip")
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(img, np.ndarray):
                img = img.swapaxes(1, 0)
                img = img[::-1, ...]
                img = img.swapaxes(0, 1)
                if lbl is not None:
                    lbl = lbl.swapaxes(1, 0)
                    lbl = lbl[::-1, ...]
                    lbl = lbl.swapaxes(0, 1)
                return img.copy(), lbl.copy()
        return img, lbl

class RandomGamma(object):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range
    def __call__(self, img, lbl, debug):
        gamma_value = random.uniform(self.gamma_range[0], self.gamma_range[1])
        if debug:
            print("Gamma: ",gamma_value)
        return Image.eval(img, lambda px: px**gamma_value), lbl

class RandomBrightness(object):
    def __init__(self, brightness_range):
        self.brightness_range = brightness_range
    def __call__(self, img, lbl, debug):
        brightness_value = random.uniform(self.brightness_range[0], self.brightness_range[1])
        if debug:
            print("Brightness: ",brightness_value)
        convert = ImageEnhance.Brightness(img)
        return convert.enhance(brightness_value), lbl

class RandomContrast(object):
    def __init__(self, contrast_range):
        self.contrast_range = contrast_range
    def __call__(self, img, lbl, debug):
        contrast_value = random.uniform(self.contrast_range[0], self.contrast_range[1])
        if debug:
            print("Contrast: ",contrast_value)
        convert = ImageEnhance.Contrast(img)
        return convert.enhance(contrast_value), lbl

class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, img, lbl, debug):
        width, height = img.size
        h_idx = random.randint(0, width - self.crop_size[0])
        w_idx = random.randint(0, height - self.crop_size[1])
        return img.crop((h_idx,w_idx, h_idx+self.crop_size[0], w_idx+self.crop_size[1])), lbl.crop((h_idx,w_idx, h_idx+self.crop_size[0], w_idx+self.crop_size[1]))

class DownsampleLabel(object):
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, img, lbl, debug):
        size = (int(lbl.size[0] / 8) , int(lbl.size[1] / 8))
        lbl.thumbnail(size, Image.NEAREST)
        return img, lbl

class ToNumpy(object):
    def __call__(self, img, lbl, debug):
        if lbl is not None:
            return np.asarray(img), np.asarray(lbl)
        else:
            return np.asarray(img), None

#img must be a numpy array
class ToFloat(object):
    def __call__(self, img, lbl, debug):    
        img2 = img.astype(float) / 255.0
        return img2.transpose(2, 0, 1), lbl

#img and lbl must be numpy arrays
class ToTensor(object):
    def __call__(self, img, lbl, debug):
        if lbl is not None:
            return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()
        else:
            return torch.from_numpy(img).float(), None


#img must be a tensor in range [0 1]
class Normalize(object):
    def __init__(self, mean_std):
        self.t = standard_transforms.Normalize(*mean_std)
    def __call__(self, img, lbl, debug):
        return self.t(img), lbl

#To call between ToNumpy and ToFloat
class SwapChannel(object):
    def __call__(self, img, lbl, debug):
        return img[:, :, [2, 1, 0]], lbl