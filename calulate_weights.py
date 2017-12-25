import tqdm
import numpy as np
from dataset.CityscapesLoader import *
from utils import *
import torch

num_classes = 19

base_data_folder = '/home/cattaneod/CITYSCAPES_crop/'

data_augmentation_train = DataAugmentation.Compose([

    DataAugmentation.ToNumpy(),
    DataAugmentation.ToFloat(),
    DataAugmentation.ToTensor(),
    ])

loader_train = CityscapesLoader2(base_data_folder, split='train',img_size=None, transforms=data_augmentation_train)

hist = torch.zeros(num_classes)

with tqdm.tqdm(loader_train, ncols=150) as t:
    for i in range(len(loader_train)):
        img, lbl = loader_train[i]
        hist += torch.histc(lbl.float(),num_classes,0,num_classes - 1)
        t.update(1)

weights = hist.median() / hist
print(weights)
