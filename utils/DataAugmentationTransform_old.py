import torchsample.transforms as transforms
import scipy.misc as m
import random
import torch
import numpy as np
import time
from utils import *
from scipy.ndimage import rotate

flip_time = AverageMeter()
rotation_time = AverageMeter()
zoom_time = AverageMeter()
trasl_time = AverageMeter()
gamma_time = AverageMeter()
bright_time = AverageMeter()
saturation_time = AverageMeter()
crop_time = AverageMeter()

def RotateScipy(image, angle, fill, spline=3):
    img = image.numpy()
    if spline>1:
        prefilter=False
    else:
        prefilter=True
    out = rotate(img, angle, reshape=False, axes=(1,2), order=spline, prefilter=prefilter, cval=fill)
    return torch.from_numpy(out)

class DataAugmentationTransform(object):
    def __init__(self,
                 split='training',
                 flip_p=0.0,
                 rotation_range = None,
                 zoom_range = None,
                 translation_range = None,
                 gamma_range = None,
                 brightness_range = None,
                 saturation_range = None,
                 crop_size = None,
                 debug = False
                 ):
        self.split = split
        self.flip_p = flip_p
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.translation_range = translation_range
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.crop_size = crop_size
        self.saturation_range = saturation_range
        self.debug = debug

    def __call__(self, image1, image2):
        if self.debug:
            #print(image1.shape)
            #print(image2.shape)
            img = image1.numpy().transpose(1, 2, 0)
            m.imsave('data_aug0.png', img)
            img2 = np.copy(image2.numpy().squeeze())
            img2 *= 10
            img2[img2 == 2550] = 255
            m.imsave('data_aug1.png', img2)
        if image2 is not None:
            image2 = image2.transpose(1,2).transpose(0,1)
        do_flip = np.random.random() < self.flip_p
        if do_flip:
            if self.debug:
                print("Flippo")
                time1 = time.process_time()
            flip = transforms.RandomFlip(p=1.0)
            image1 = flip(image1)
            if image2 is not None:
                image2 = flip(image2)
            if self.debug:
                time2 = time.process_time()
                flip_time.update(time2-time1)
                print("Average Flip Time: ",flip_time.avg)
        if self.gamma_range:
            gamma_value = random.uniform(self.gamma_range[0], self.gamma_range[1])
            if self.debug:
                print("Gamma :", gamma_value)
                time1 = time.process_time()
            gamma_transform = transforms.Gamma(gamma_value)
            image1 = gamma_transform(image1)
            if self.debug:
                time2 = time.process_time()
                gamma_time.update(time2-time1)
                print("Average Gamma Time: ",gamma_time.avg)
        if self.brightness_range:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            if self.debug:
                print("Brightness :", brightness)
                time1 = time.process_time()
            brightness_transform = transforms.Brightness(brightness)
            image1 = brightness_transform(image1)
            if self.debug:
                time2 = time.process_time()
                bright_time.update(time2-time1)
                print("Average Brightness Time: ",bright_time.avg)
        if self.saturation_range:
            saturation = random.uniform(self.saturation_range[0], self.saturation_range[1])
            if self.debug:
                print("Saturation :", saturation)
                time1 = time.process_time()
            saturation_transform = transforms.Saturation(saturation)
            image1 = saturation_transform(image1)
            if self.debug:
                time2 = time.process_time()
                saturation_time.update(time2-time1)
                print("Average Saturation Time: ",saturation_time.avg)
        if self.translation_range:
            height_range = self.translation_range[0]
            width_range = self.translation_range[1]
            random_height = random.uniform(-height_range, height_range)
            random_width = random.uniform(-width_range, width_range)
            if self.debug:
                print("Traslo di: ",random_height," , ",random_width)
                time1 = time.process_time()
            translate_input = transforms.Translate([random_height, random_width], interp='bilinear')
            translate_target = transforms.Translate([random_height, random_width], interp='nearest')
            image1 = translate_input(image1)
            if image2 is not None:
                image2 = translate_target(image2)
            if self.debug:
                time2 = time.process_time()
                trasl_time.update(time2-time1)
                print("Average Transl Time: ",trasl_time.avg)
        if self.zoom_range:
            zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
            zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
            if self.debug:
                print("Zoommo di: ", zx, " , ", zy)
                time1 = time.process_time()
            zoom_input = transforms.Zoom([zx, zy], interp='bilinear')
            zoom_target = transforms.Zoom([zx, zy], interp='nearest')
            image1 = zoom_input(image1)
            if image2 is not None:
                image2 = zoom_target(image2)
            if self.debug:
                time2 = time.process_time()
                zoom_time.update(time2-time1)
                print("Average zoom Time: ",zoom_time.avg)
        if self.rotation_range:
            degree = random.uniform(-self.rotation_range, self.rotation_range)
            if self.debug:
                print("Ruoto di: ",degree)
                time1 = time.process_time()
            image1 = RotateScipy(image1, degree,fill=0.0, spline=2, )
            if image2 is not None:
                image2 = RotateScipy(image2, degree, fill=255, spline=0 )

            #rotate_input = transforms.Rotate(degree, interp='bilinear')
            #rotate_target = transforms.Rotate(degree, interp='nearest')
            #image1 = rotate_input(image1)
            #if image2 is not None:
            #    image2 = rotate_target(image2)
            if self.debug:
                time2 = time.process_time()
                rotation_time.update(time2-time1)
                print("Average Rotate Time: ",rotation_time.avg)
        if self.crop_size:
            time1 = time.process_time()
            h_idx = random.randint(0,image1.size(1)-self.crop_size[0])
            w_idx = random.randint(0,image1.size(2) - self.crop_size[1])
            image1 = image1[:, h_idx:(h_idx + self.crop_size[0]), w_idx:(w_idx + self.crop_size[1])]
            if image2 is not None:
                image2 = image2[:, h_idx:(h_idx + self.crop_size[0]), w_idx:(w_idx + self.crop_size[1])]
            if self.debug:
                time2 = time.process_time()
                crop_time.update(time2-time1)
                print("Average Crop Time: ",crop_time.avg)

        if image2 is not None:
            image2 = image2.transpose(0,1).transpose(1,2)
        if self.debug:
            #print(image1.shape)
            img = image1.numpy().transpose(1,2,0)
            m.imsave('data_aug2.png',img)
            #print(image2.shape)
            img2 = np.copy(image2.numpy().squeeze())
            img2 *= 10
            img2[img2 == 2550] = 255
            m.imsave('data_aug3.png',img2)

        if image2 is not None:
            image2 = image2.type(torch.LongTensor)
        return image1, image2

