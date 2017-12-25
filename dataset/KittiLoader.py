import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from glob import glob
import os.path
import re
import random

from tqdm import tqdm
from torch.utils import data


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']


class KittiLoader(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=512, transforms=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms

        for split in ["training", "testing", "validation"]:
            file_list = glob(os.path.join(root, split, 'image_2', '*.png'))
            self.files[split] = file_list
            label_list = {
                re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                for path in glob(os.path.join(root, split, 'gt_image_2', '*_road_*.png'))}
            self.labels[split] = label_list

        '''
        if not os.path.isdir(self.root + '/SegmentationClass/pre_encoded'):
            self.setup(pre_encode=True)
        else:
            self.setup(pre_encode=False)
        '''

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = img_name

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.split != "testing":
            lbl_path = self.labels[self.split][os.path.basename(img_path)]
            lbl = m.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.int32)
        else:
            lbl = None

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        if self.transforms:
            img, lbl = self.transforms(img, lbl)

        if self.split != "testing":
            return img, lbl
        else:
            return img

    def encode_segmap(self, mask):
        background_color = np.array([255, 0, 0])
        gt_bg = np.all(mask == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        return gt_bg

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        img -= self.mean
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        if self.split != "testing":
            lbl = lbl.astype(float)
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), interp='nearest')
            lbl = self.encode_segmap(lbl)
            lbl = lbl.astype(int)
            lbl = torch.from_numpy(lbl).long()
        else:
            lbl = None

        img = torch.from_numpy(img).float()
        return img, lbl

    def get_pascal_labels(self):
        return np.asarray([[1.,1.,1.], [0., 0., 0.]])

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
''''
    def setup(self, pre_encode=False):
        sbd_path = get_data_path('sbd')
        voc_path = get_data_path('pascal')

        target_path = self.root + '/SegmentationClass/pre_encoded/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        sbd_train_list = tuple(open(sbd_path + 'dataset/train.txt', 'r'))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]

        self.files['train_aug'] = self.files['train'] + sbd_train_list

        if pre_encode:
            print
            "Pre-encoding segmentation masks..."
            for i in tqdm(sbd_train_list):
                lbl_path = sbd_path + 'dataset/cls/' + i + '.mat'
                lbl = io.loadmat(lbl_path)['GTcls'][0]['Segmentation'][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i + '.png', lbl)

            for i in tqdm(self.files['trainval']):
                lbl_path = self.root + '/SegmentationClass/' + i + '.png'
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i + '.png', lbl)
'''

if __name__ == '__main__':
    local_path = '/home/cattaneod/CarND-Semantic-Segmentation/data/data_road'
    dst = KittiLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4, shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            labels = labels.numpy()[0, :, :, 0]
            labels = dst.decode_segmap(labels)

    print("CIAO")
    for i, data in enumerate(trainloader):
        imgs, labels = data
        break