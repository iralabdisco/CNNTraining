import collections
import json
import os
import os.path
import re
from collections import namedtuple
from glob import glob
import numpy as np
import scipy.misc as m
import torch

import matplotlib.pyplot as plt
from torch.utils import data

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         ,[  0,  0,  0] ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         ,[  0,  0,  0] ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         ,[  0,  0,  0] ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         ,[  0,  0,  0] ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         ,[  0,  0,  0] ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         ,[111, 74,  0] ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         ,[ 81,  0, 81] ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        ,[128, 64,128] ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        ,[244, 35,232] ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         ,[250,170,160] ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         ,[230,150,140] ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        ,[ 70, 70, 70] ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        ,[102,102,156] ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        ,[190,153,153] ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         ,[180,165,180] ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         ,[150,100,100] ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         ,[150,120, 90] ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        ,[153,153,153] ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         ,[153,153,153] ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        ,[250,170, 30] ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        ,[220,220,  0] ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        ,[107,142, 35] ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        ,[152,251,152] ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        ,[ 70,130,180] ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        ,[220, 20, 60] ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        ,[255,  0,  0] ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        ,[  0,  0,142] ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        ,[  0,  0, 70] ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        ,[  0, 60,100] ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         ,[  0,  0, 90] ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         ,[  0,  0,110] ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        ,[  0, 80,100] ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        ,[  0,  0,230] ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        ,[119, 11, 32] ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         ,[  0,  0,142] ),
]

def get_id2color():
    return {label.trainId : label.color for label in labels}

def get_trainId2Id():
    return {label.trainId : label.id for label in labels if label.ignoreInEval == False}


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']


class CityscapesLoader(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=None, transforms=None, augmentation=1, return_original = False):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.augmentation = augmentation
        self.return_original = return_original
        if img_size:
            self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        else:
            self.img_size = None
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms
        self.trainId2color = {label.trainId : label.color for label in labels}

        for split in ["train", "test", "val"]:
            file_list = glob(root + 'leftImg8bit/' + split + '/**/*.png', recursive=True)
            self.files[split] = file_list
            label_list = {
                os.path.basename(path): re.sub('_leftImg8bit.', '_gtFine_labelTrainIds.', re.sub('leftImg8bit/', 'gtFine/', path))
                for path in file_list}
            self.labels[split] = label_list

    def __len__(self):
        return self.augmentation * len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][int(index / self.augmentation)]
        img_path = img_name

        img = m.imread(img_path)

        assert isinstance(img,np.ndarray)
        #print(type(img[0][0][0]))
        #img = np.array(img, dtype=np.uint8)
        img = img.astype(np.uint8)
        if self.return_original:
            original = np.copy(img)
            original = self.transform_original(original)
            base_name = os.path.basename(img_path)

        if self.split not in ["testing", "test"]:
            lbl_path = self.labels[self.split][os.path.basename(img_path)]
            lbl = m.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)
        else:
            lbl = None

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        if self.transforms:
            img, lbl = self.transforms(img, lbl)

        if self.split not in ["testing","test"]:
            if self.return_original:
                return base_name, original, img, lbl
            else:
                return img, lbl
        else:
            if self.return_original:
                return base_name, original, img
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
        if self.img_size:
            img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        #img -= self.mean
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        if self.split not in ["testing", "test"]:
            lbl = lbl.astype(float)
            if self.img_size:
                lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), interp='nearest')
            lbl = np.expand_dims(lbl, 2)
            lbl = lbl.astype(int)
            lbl = torch.from_numpy(lbl).long()
        else:
            lbl = None

        img = torch.from_numpy(img).float()
        return img, lbl

    def transform_original(self, img):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def get_pascal_labels(self):
        return self.trainId2color

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

if __name__ == '__main__':
    from utils.DataAugmentationTransform_old import *
    num_classes = 19
    image_shape = (1024, 2048)
    epochs = 30
    batch_size = 4
    l_rate = 0.0001
    freeze_layers = False
    resume = False
    save = True
    check_validation = False
    overlay_during_training = True
    base_save_folder = "fcn1s"
    base_data_folder = '/home/cattaneod/CITYSCAPES/'
    resume_filename = "./saved/" + base_save_folder + "/checkpoint_78_0.910070304488.pth.tar"
    data_augmentation = DataAugmentationTransform_old(crop_size=(500, 500), debug=True,
                                                      translation_range=(0.0,0.15),
                                                      rotation_range=10,
                                                      zoom_range = (0.8, 1.0),
                                                      flip_p = 0.5,
                                                      brightness_range = (-0.2, 0.2),
                                                      gamma_range = (0.5, 1.5),
                                                      saturation_range=(-0.3, 0.3))
    loader_train = CityscapesLoader(base_data_folder, split='train', is_transform=True, img_size=image_shape,
                                    transforms=data_augmentation)
    loader_train[0]