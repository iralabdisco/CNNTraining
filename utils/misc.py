import numpy as np
from sklearn import metrics
import torch.nn.functional as F
from torch import nn
import torch
from torch.autograd import Variable
from collections import deque
import time


class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

class AverageMeter(object):
    def __init__(self, moving_average=None):
        self.max = moving_average
        self.deque = deque()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.deque.clear()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.max is not None:
            self.deque.append(val * n)
            if self.count > self.max:
                pop = self.deque.popleft()
                self.sum -= pop
                self.count -= n
        self.avg = self.sum / self.count


def accuracy_IoU_CFMatrix(input, target, classes):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2)
    log_p = log_p.transpose(2, 3)
    log_p = log_p.contiguous()
    log_p = log_p.view(-1, c)

    topk=(1,)
    maxk = max(topk)
    _, pred = log_p.topk(maxk, 1, True, True)
    pred = pred.data.cpu().numpy()
    pred = pred.transpose()[0]

    mask = target >= 0
    target = target[mask]
    target = target.data.cpu().numpy()
    try:
        cf_matrix = metrics.confusion_matrix(y_pred=pred, y_true=target, labels=classes)
        rows = cf_matrix.sum(axis=1)
        cols = cf_matrix.sum(axis=0)
        IoU = np.ndarray(cf_matrix.shape[0])
        for i in range(cf_matrix.shape[0]):
            if rows[i] + cols[i] > 0.:
                IoU[i] = cf_matrix[i][i] / (rows[i] + cols[i] - cf_matrix[i][i])
            else:
                IoU[i] = np.nan

        return np.trace(cf_matrix) / np.sum(cf_matrix), IoU, cf_matrix
    except ValueError:
        return None, None, None

def accuracy_IoU(input, target, classes):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2)
    log_p = log_p.transpose(2, 3)
    log_p = log_p.contiguous()
    log_p = log_p.view(-1, c)

    topk=(1,)
    maxk = max(topk)
    _, pred = log_p.topk(maxk, 1, True, True)
    pred = pred.data.cpu().numpy()
    pred = pred.transpose()[0]

    mask = target >= 0
    target = target[mask]
    target = target.data.cpu().numpy()
    try:
        accuracy = metrics.accuracy_score(y_true=target, y_pred=pred)
        IoU = metrics.jaccard_similarity_score(y_true=target, y_pred=pred)

        return accuracy, IoU
    except ValueError:
        return None, None

def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index = -100):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2)
    log_p = log_p.transpose(2, 3)
    log_p = log_p.contiguous()
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=ignore_index)
    if size_average:
        loss /= mask.data.sum()
    del log_p
    del mask
    return loss

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    new_lr = init_lr*(1 - iter/max_iter)**power
    #print("New LR: ", new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def poly_lr2(init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return None

    return init_lr*(1 - float(iter)/max_iter)**power



def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class Conv2dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x

class SlidedForward(object):
    def __init__(self, forward, tile_horizontal, tile_vertical, crop_size, image_size, transforms, softmax = False):
        self.forward = forward
        self.tile_horizontal = tile_horizontal
        self.tile_vertical = tile_vertical
        self.crop_size = crop_size
        self.image_size = image_size
        self.stride_horizontal = int((image_size[2] - crop_size) / (tile_horizontal - 1))
        self.stride_vertical = int((image_size[1] - crop_size) / (tile_vertical - 1))
        self.step_horizontal = [i * self.stride_horizontal for i in range(tile_horizontal)]
        self.step_vertical = [i * self.stride_vertical for i in range(tile_vertical)]
        self.step_horizontal[-1] = (image_size[2] - crop_size)
        self.step_vertical[-1] = (image_size[1] - crop_size)
        self.transforms = transforms
        self.softmax = softmax


    def slided_forward(self, image):
        predicted_img = np.zeros(self.image_size, dtype=np.float32)
        count_img = np.zeros(self.image_size, dtype=np.uint8)
        for shift_lateral in self.step_horizontal:
            for shift_vertical in self.step_vertical:
                #print("Crop: %d %d %d %d" % (shift_vertical, shift_vertical + self.crop_size, shift_lateral , shift_lateral + self.crop_size))
                cropped_img = image[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size]
                cropped_img, _ = self.transforms(cropped_img, None)
                cropped_img = cropped_img.unsqueeze(0)
                if torch.cuda.is_available():
                    cropped_img = Variable(cropped_img.cuda(0), volatile=True)
                else:
                    cropped_img = Variable(cropped_img, volatile=True)
                cropped_pred = self.forward(cropped_img)
                if self.softmax:
                    cropped_pred = F.log_softmax(cropped_pred)
                predicted_img[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size] += cropped_pred.cpu().data.numpy()[0]
                count_img[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size] += 1

        return predicted_img #/ count_img

#THIS IS NOT WORKING!!
class SlidedForwardBatch(object):
    def __init__(self, forward, tile_horizontal, tile_vertical, crop_size, image_size, transforms, batch_size, softmax = False):
        self.forward = forward
        self.tile_horizontal = tile_horizontal
        self.tile_vertical = tile_vertical
        self.crop_size = crop_size
        self.image_size = image_size
        self.stride_horizontal = int((image_size[2] - crop_size) / (tile_horizontal - 1))
        self.stride_vertical = int((image_size[1] - crop_size) / (tile_vertical - 1))
        self.step_horizontal = [i * self.stride_horizontal for i in range(tile_horizontal)]
        self.step_vertical = [i * self.stride_vertical for i in range(tile_vertical)]
        self.step_horizontal[-1] = (image_size[2] - crop_size)
        self.step_vertical[-1] = (image_size[1] - crop_size)
        self.transforms = transforms
        self.batch_size = batch_size
        self.softmax = softmax


    def slided_forward(self, image):
        predicted_img = np.zeros(self.image_size, dtype=np.float32)
        count_img = np.zeros(self.image_size, dtype=np.uint8)
        splitted_batch = torch.zeros((self.tile_horizontal * self.tile_vertical, 3, self.crop_size, self.crop_size))
        batch_pred =  torch.zeros((self.tile_horizontal * self.tile_vertical, self.image_size[0], self.crop_size, self.crop_size))
        i = 0
        for shift_lateral in self.step_horizontal:
            for shift_vertical in self.step_vertical:
                #print("Crop: %d %d %d %d" % (shift_vertical, shift_vertical + self.crop_size, shift_lateral , shift_lateral + self.crop_size))
                cropped_img = image[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size]
                cropped_img, _ = self.transforms(cropped_img, None)
                splitted_batch[i] = cropped_img.clone()
                i += 1

        for i in range(int(self.tile_horizontal * self.tile_vertical / self.batch_size)):
                if torch.cuda.is_available():
                    cropped_img = Variable(splitted_batch[i*self.batch_size:(i+1)*self.batch_size].cuda(0), volatile=True)
                else:
                    cropped_img = Variable(splitted_batch[i*self.batch_size:(i+1)*self.batch_size], volatile=True)
                batch_pred[i*self.batch_size:(i+1)*self.batch_size] = self.forward(cropped_img).cpu().data.clone()

        i = 0
        for shift_lateral in self.step_horizontal:
            for shift_vertical in self.step_vertical:
                if self.softmax:
                    batch_pred = F.log_softmax(batch_pred)
                predicted_img[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size] += batch_pred[i].numpy()[0]
                count_img[:, shift_vertical:shift_vertical + self.crop_size, shift_lateral : shift_lateral + self.crop_size] += 1
                i += 1

        return predicted_img #/ count_img

