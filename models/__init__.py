import torchvision.models as models

from models.fcn import *
import numpy as np
from sklearn import metrics



def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s', 'fcn1s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'fcn1s' : fcn1s
    }[name]