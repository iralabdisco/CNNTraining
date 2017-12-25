import tqdm
from PIL import Image
from graphviz import Digraph
from tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np
from torch.backends import cudnn
from dataset.CityscapesLoader import *
from models import erfnet
from utils import *
from torch.optim import SGD, Adam, lr_scheduler

cudnn.benchmark = True
cudnn.enabled = True

total_inference = 0

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def tensor_to_imgs(input, output, target=None):
    log_p = output
    log_p = log_p.transpose(1, 2)
    log_p = log_p.transpose(2, 3)
    out = log_p.data.cpu().numpy()
    out = np.argmax(out, axis = 3)
    '''
    mask = 1 - out
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    mask = np.dot(mask, np.array([[0, 255, 0, 127]]))
    '''
    mask = np.copy(out)
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    mask = np.dot(mask, np.array([[0, 1, 1, 1]]))
    id2color = get_id2color()
    id2color[19] = [0,0,0]
    for i in range(num_classes):
        mask[out==i] = np.concatenate((id2color[i], [127]))
    mask = mask.astype(np.uint8)

    out = out.astype(np.uint8)
    if target:
        lab = target.numpy()
        lab = lab.squeeze()
        lab = lab * 255
        lab = lab.astype(np.uint8)
    imgs = input.transpose(1,2)
    imgs = imgs.transpose(2,3)
    imgs = imgs.data.cpu().numpy()
    imgs *= np.array([0.229, 0.224, 0.225])
    imgs += np.array([0.485, 0.456, 0.406])
    imgs = (imgs * 255)
    imgs = imgs.astype('uint8')
    return mask, imgs, out

def numpy_to_imgs(input, output, target=None):
    log_p = output
    log_p = log_p.transpose(0,2,3,1)
    out = np.argmax(log_p, axis=3)
    '''
    mask = 1 - out
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    mask = np.dot(mask, np.array([[0, 255, 0, 127]]))
    '''
    mask = np.copy(out)
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    mask = np.dot(mask, np.array([[0, 1, 1, 1]]))
    id2color = get_id2color()
    id2color[19] = [0,0,0]
    for i in range(num_classes):
        mask[out == i] = np.concatenate((id2color[i], [127]))
    mask = mask.astype(np.uint8)

    out = out.astype(np.uint8)
    if target:
        lab = target.numpy()
        lab = lab.squeeze()
        lab = lab * 255
        lab = lab.astype(np.uint8)
    imgs = input.transpose(0,2,3,1)
    #imgs *= np.array([0.229, 0.224, 0.225])
    #imgs += np.array([0.485, 0.456, 0.406])
    imgs = (imgs * 255)
    imgs = imgs.astype('uint8')
    return mask, imgs, out


def overlay_images(names, input, output, starting_index, runs="", tensor=True, convert_id = False):
    if tensor:
        mask, imgs, out = tensor_to_imgs(input, output)
    else:
        mask, imgs, out = numpy_to_imgs(input, output)
    for i in range(mask.shape[0]):
        temp = np.copy(out[i])
        if convert_id:
            trainId2Id = get_trainId2Id()
            for j in range(len(trainId2Id)):
                temp[out[i] == j] = trainId2Id[j]
                res = Image.fromarray(temp)
                res.save('/home/cattaneod/CITYSCAPES/results/'+names[i])

        m = Image.fromarray(mask[i])
        im = Image.fromarray(imgs[i])
        im.paste(m, box=None, mask=m)
        im.save('./CITYSCAPES/ERF/inference/'+str(runs)+str(starting_index + i)+".png")


num_classes = 20
image_shape = (512,1024)
epochs = 20
batch_size = 1
l_rate = 5e-4
weight_decay = 2e-4
update_batches = 10

freeze_layers = False
resume = False
save = True
check_validation = True
overlay_during_training = True
poly_lr = True
TBWriter = True
use_weights = True
doIouOrig = True
slided_prediction = False

moving_average = 1000
checkpoint_save = 1000
TBUpdate = 100
num_workers=2

base_save_folder = "./CITYSCAPES/ERF/models"
base_data_folder = '/home/cattaneod/CITYSCAPES/'
#resume_filename = "./CITYSCAPES/models/checkpoint_17_0.335076503344.pth.tar"
resume_filename = "models/pretrained/erfnet_pretrained.pth"
opt = "Adam"

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_augmentation_train = DataAugmentation.Compose([
    DataAugmentation.RandomHorizontallyFlip(0.5),
    DataAugmentation.RandomBrightness((0.9, 1.1)),
    DataAugmentation.RandomGamma((0.9, 1.1)),
    DataAugmentation.RandomBrightness((0.9, 1.1)),
    DataAugmentation.RandomContrast((0.9, 1.1)),
    DataAugmentation.RandomRotate(10),
    #DataAugmentation.RandomCrop((512, 512)),

    DataAugmentation.ToNumpy(),
    DataAugmentation.ToFloat(),
    DataAugmentation.ToTensor(),
    DataAugmentation.Normalize(mean_std)
    ])

data_augmentation_val = DataAugmentation.Compose([
    #DataAugmentation.RandomCrop((512, 512)),
    DataAugmentation.ToNumpy(),
    DataAugmentation.ToFloat(),
    DataAugmentation.ToTensor(),
    DataAugmentation.Normalize(mean_std)
    ])


def train():
    if use_weights:
        weight = torch.ones(num_classes)
        '''
        #The following wheigts are taken from https://github.com/Eromera/erfnet_pytorch/blob/master/train/main.py
        weight[0] = 2.8149201869965
        weight[1] = 6.9850029945374
        weight[2] = 3.7890393733978
        weight[3] = 9.9428062438965
        weight[4] = 9.7702074050903
        weight[5] = 9.5110931396484
        weight[6] = 10.311357498169
        weight[7] = 10.026463508606
        weight[8] = 4.6323022842407
        weight[9] = 9.5608062744141
        weight[10] = 7.8698215484619
        weight[11] = 9.5168733596802
        weight[12] = 10.373730659485
        weight[13] = 6.6616044044495
        weight[14] = 10.260489463806
        weight[15] = 10.287888526917
        weight[16] = 10.289801597595
        weight[17] = 10.405355453491
        weight[18] = 10.138095855713
        '''

        #The following weights are calculated using calculate_weights.py (hist.median() / hist)
        weight[0] = 0.0238
        weight[1] = 0.1540
        weight[2] = 0.0447
        weight[3] = 1.3481
        weight[4] = 1.0000
        weight[5] = 0.7090
        weight[6] = 4.6042
        weight[7] = 1.6716
        weight[8] = 0.0622
        weight[9] = 0.7796
        weight[10] = 0.3195
        weight[11] = 0.6157
        weight[12] = 5.2630
        weight[13] = 0.1177
        weight[14] = 3.0565
        weight[15] = 3.2344
        weight[16] = 3.4215
        weight[17] = 8.1690
        weight[18] = 1.9417

    else:
        weight = None

    loader_train = CityscapesLoader2(base_data_folder, split='train',img_size=image_shape, transforms=data_augmentation_train)
    trainloader = data.DataLoader(loader_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = erfnet.ERFNet(num_classes)
    model.cuda()
    weight = weight.cuda()

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    start_epoch = 1
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / 10)), 0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(0, 10):
        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)
        epoch_loss = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        for step, (images, labels) in enumerate(trainloader):
            images = images.cuda()
            labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, only_encode=True)
            optimizer.zero_grad()
            main_loss = misc.cross_entropy2d(outputs, targets, weight=weight, ignore_index=255)
            main_loss.backward()
            optimizer.step()
            epoch_loss.append(main_loss.data[0])
            print("Loss: ", main_loss )

train()



