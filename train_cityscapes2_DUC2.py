import tqdm
from PIL import Image
from graphviz import Digraph
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import numpy as np
from torch.backends import cudnn
from dataset.CityscapesLoader import *
from models import duc_hdc
from utils import *

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
                res.save('/home/cattaneod/Datasets/CITYSCAPES/results/'+names[i])

        m = Image.fromarray(mask[i])
        im = Image.fromarray(imgs[i])
        im.paste(m, box=None, mask=m)
        im.save('./CITYSCAPES/DUC2/inference/'+str(runs)+str(starting_index + i)+".png")


num_classes = 19
image_shape = (256,512)
epochs = 20
batch_size = 8
l_rate = 2.5e-4
weight_decay = 5e-4
update_batches = 1

freeze_layers = False
resume = True
save = True
check_validation = True
overlay_during_training = False
poly_lr = True
TBWriter = True
use_weights = False
doIouOrig = True
doTrainStats = False
doCFMatrixTrain = False

moving_average = 5000
checkpoint_save = int(1000 / batch_size) * batch_size
TBUpdate = 100
num_workers=6

base_save_folder = "./CITYSCAPES/DUC2/models"
base_data_folder = '/home/cattaneod/Datasets/CITYSCAPES_crop/'
#resume_filename = "./CITYSCAPES/models/checkpoint_17_0.335076503344.pth.tar"
resume_filename = base_save_folder + "/checkpoint_8_0.3320905423883114.pth.tar"
opt = "SGD"

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_augmentation_train = DataAugmentation.Compose([
    DataAugmentation.RandomRotate(10),
    DataAugmentation.RandomHorizontallyFlip(0.5),
    #DataAugmentation.RandomBrightness((0.9, 1.1)),
    #DataAugmentation.RandomGamma((0.9, 1.1)),
    #DataAugmentation.RandomBrightness((0.9, 1.1)),
    #DataAugmentation.RandomContrast((0.9, 1.1)),
    DataAugmentation.RandomCrop((512, 512)),

    DataAugmentation.ToNumpy(),
    DataAugmentation.ToFloat(),
    DataAugmentation.ToTensor(),
    DataAugmentation.Normalize(mean_std)
    ])

data_augmentation_val = DataAugmentation.Compose([
    DataAugmentation.RandomCrop((512, 512)),
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

    loader_train = CityscapesLoader2(base_data_folder, split='train',img_size=None, transforms=data_augmentation_train)
    trainloader = data.DataLoader(loader_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    #loader_test = CityscapesLoader2(base_data_folder, split='test', is_transform=True, img_size=None, transforms=data_augmentation)
    #test_loader = data.DataLoader(loader_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    loader_val = CityscapesLoader2(base_data_folder, split='val', img_size=None, transforms=data_augmentation_val)
    valloader = data.DataLoader(loader_val, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    model = duc_hdc.ResNetDUCHDC(num_classes)

    if TBWriter:
        writer = SummaryWriter('./runs/HDC_DUC/')

    '''
    if resume:
        print("Loading from: ", resume_filename)
        saved_state_dict = torch.load(resume_filename)
        if num_classes != 21:
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]

        model.load_state_dict(saved_state_dict)
    '''

    if torch.cuda.is_available():
        print("Using GPU")
        model.cuda(0)
        if use_weights:
            weight = weight.cuda()
    else:
        print("Using CPU")

    model.train()
    gpus = [0, 1, 2, 3]
    model = torch.nn.DataParallel(model, device_ids=gpus)

    if freeze_layers:
        print("Freezing ResNet layers")
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if opt == "SGD":
        optimizer = torch.optim.SGD(parameters, l_rate, momentum=0.9, weight_decay=weight_decay)
    elif opt =="Adam":
        optimizer = torch.optim.Adam(parameters, l_rate, weight_decay=weight_decay)

    starting_epoch = 0
    starting_iteration = 0
    lr_ = l_rate

    if resume:
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        starting_iteration = int(checkpoint['iter'] % 35700 / batch_size)
        print("Startin epoch: "+str(starting_epoch)+", starting iter: ",str(starting_iteration))
        if poly_lr:
            lr_ = poly_lr2(l_rate, starting_iteration + len(trainloader) * starting_epoch, lr_decay_iter=1, max_iter=len(trainloader) * epochs)
            if lr_:
                optimizer.param_groups[0]['lr'] = lr_
        model.load_state_dict(saved_state_dict)


    best_metric = 0
    old_file = ""
    old_checkpoint = ""
    train_acc = AverageMeter()
    train_IoU = AverageMeter()
    train_loss = AverageMeter()
    local_acc = AverageMeter(moving_average=moving_average)
    local_IoU = AverageMeter(moving_average=moving_average)
    local_loss = AverageMeter(moving_average=moving_average)
    for epoch in range(starting_epoch, epochs):
        train_acc.reset()
        train_IoU.reset()
        train_loss.reset()
        train_cfmatrix = np.zeros((num_classes, num_classes))

        print("\nEpoch: ",epoch)

        if overlay_during_training and epoch % 1  == 0:
            for i in range(15):
                print("Overlaying image ",i)
                test_img, _ = loader_val[i]
                test_img = test_img.unsqueeze(0)
                #original_img = original_img.unsqueeze(0)
                #original_img = Variable(original_img.cuda())
                model.eval()
                test_pred = model(Variable(test_img.cuda(0), requires_grad=True))
                test_img = Variable(test_img.cuda(0), requires_grad=True)
                #if TBWriter and i==0:
                #    writer.add_graph(model, test_pred)
                #test_pred = F.upsample_bilinear(test_pred, (1024, 2048))
                overlay_images('', test_img, test_pred, epoch, str(i) + '_', convert_id=False)
                del test_pred
                del test_img

        model.train()
        optimizer.zero_grad()
        with tqdm.tqdm(trainloader, ncols=150) as t:
            if epoch == starting_epoch:
                t.update(starting_iteration)
            for i, (images, labels) in enumerate(t):
                if torch.cuda.is_available():
                    images = Variable(images).cuda(0)
                    labels = Variable(labels).cuda(0)
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                iteration = len(trainloader) * epoch + i
                processed_image = i * batch_size
                if epoch == starting_epoch:
                    iteration += starting_iteration
                    processed_image += starting_iteration * batch_size


                outputs = model(images)
                #g = make_dot(outputs)
                #g.save('./t.dot')

                main_loss = misc.cross_entropy2d(outputs, labels,weight=weight, ignore_index=255)

                loss = main_loss
                loss = loss / update_batches

                loss.backward()

                t.set_description('Loss: %8.4f - LR = %f' % (update_batches * loss.data[0], lr_))

                train_loss.update(update_batches * loss.data[0])
                local_loss.update(update_batches * loss.data[0])
                if doTrainStats:
                    if doCFMatrixTrain:
                        acc, IoU, cf_matrix =  accuracy_IoU_CFMatrix(outputs,labels, np.array(range(num_classes)))
                        IoU = IoU.mean
                    else:
                        acc, IoU =  accuracy_IoU(outputs,labels, np.array(range(num_classes)))

                    if acc is not None:
                        train_acc.update(acc)
                        train_IoU.update(IoU)
                        local_acc.update(acc)
                        local_IoU.update(IoU)
                        if doCFMatrixTrain:
                            train_cfmatrix = train_cfmatrix + cf_matrix

                if i % update_batches == 0:
                    optimizer.step()
                    if poly_lr:
                        lr_ = poly_lr2(l_rate, iteration, lr_decay_iter=1, max_iter=len(trainloader) * epochs)
                        if lr_:
                            t.set_description('Step: %8.4f - LR = %f' % (update_batches * loss.data[0], lr_))
                            optimizer.param_groups[0]['lr'] = lr_

                    #print("%8.2f %%  ->  Loss: %8.6f " % (i / len(trainloader) * 100, loss.data[0]), end='\r')
                    optimizer.zero_grad()

                if local_loss.count > int(500 / batch_size) and processed_image % TBUpdate == 0 and TBWriter:
                    if doTrainStats:
                        writer.add_scalar('Train Accuracy', local_acc.avg, iteration * batch_size)
                        writer.add_scalar('Train IoU', local_IoU.avg, iteration * batch_size)
                    writer.add_scalar('Train Loss', local_loss.avg, iteration * batch_size)

                del outputs
                del loss
                del images
                del labels

                if i > 0 and local_loss.count > int(500 / batch_size) and processed_image % checkpoint_save == 0:
                    save_name = base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(processed_image) + "_" + str(local_loss.avg) + ".pth.tar"
                    torch.save({
                        'epoch': epoch,
                        'iter' : processed_image,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                        save_name)
                    #print("Model Saved As " + save_name)
                    if os.path.isfile(old_checkpoint):
                        os.remove(old_checkpoint)
                    old_checkpoint = save_name

                t.update(1)

                if i+starting_iteration+1 == len(trainloader):
                    break


        if doCFMatrixTrain and doTrainStats:
            rows = train_cfmatrix.sum(axis=1)
            cols = train_cfmatrix.sum(axis=0)
            IoU = np.ndarray(train_cfmatrix.shape[0])
            for i in range(train_cfmatrix.shape[0]):
                if rows[i] + cols[i] > 0.:
                    IoU[i] = train_cfmatrix[i][i] / (rows[i] + cols[i] - train_cfmatrix[i][i])
                else:
                    IoU[i] = np.nan
            print("Macro IoU: ", np.nanmean(IoU), "\n")
            print("Macro Accuracy: ", np.trace(train_cfmatrix) / np.sum(train_cfmatrix))

        if doTrainStats:
            print("\nMicro Accuracy: ", train_acc.avg)
            print("Micro IoU: ", train_IoU.avg, "\n")
        print("Train Loss: ", train_loss.avg)

        if check_validation:
            val_IoU = eval(model)
            if TBWriter:
                writer.add_scalar('Val IoU', val_IoU, epoch)

        if doCFMatrixTrain:
            save_metric = np.nanmean(IoU)
        elif doTrainStats:
            save_metric = train_IoU.avg
        else:
            save_metric = train_loss.avg
        if check_validation and doIouOrig:
            save_metric = val_IoU

        if best_metric < save_metric:
            best_metric = save_metric
            print("New Best IoU!")
            if save:
                torch.save({
                    'epoch': epoch + 1,
                    'iter' : 0,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                 base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar")
                print("Model Saved As " + base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar")
                if os.path.isfile(old_file):
                    os.remove(old_file)
                old_file = base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar"

        print("Best IoU So Far: ", best_metric)

    if TBWriter:
        writer.close()
    print("End Of Training")


from datetime import datetime

def eval(model):
    confMatrix = generateMatrixTrainId(evalIoU.args)
    perImageStats = {}
    nbPixels = 0
    iouVal = None

    data_augmentation_slided = DataAugmentation.Compose([
        DataAugmentation.ToNumpy(),
        DataAugmentation.ToFloat(),
    ])
    data_augmentation_slided2 = DataAugmentation.Compose([
        DataAugmentation.ToTensor(),
        DataAugmentation.Normalize(mean_std)
    ])

    loader_val = CityscapesLoader2('/home/cattaneod/Datasets/CITYSCAPES/', split='val', img_size=None, transforms=data_augmentation_slided, return_name=True)

    slided_forward = SlidedForward(model, 6, 5, 512, (num_classes, 1024, 2048), data_augmentation_slided2, softmax=False)

    model.eval()
    #runs = datetime.now().strftime('%b%d_%H-%M-%S')+"/"
    #os.makedirs('./CITYSCAPES/DUC2/inference/last/')
    mean_time = AverageMeter()
    with tqdm.tqdm(loader_val, ncols=150) as t:
        for i in range(len(loader_val)):

            name, image, lbl = loader_val[i]
            t1 = time.time()
            predicted_img = slided_forward.slided_forward(image)
            t2 = time.time()
            mean_time.update(t2-t1)
            image = np.expand_dims(image, axis=0)
            predicted_img = np.expand_dims(predicted_img, axis=0)
            overlay_images([name], image, predicted_img, i, 'last/' , tensor=False, convert_id=True)
            t.set_description('Prediction Time: %8.4f' % (mean_time.avg))
            t.update(1)

            if doIouOrig:
                prediction = np.argmax(predicted_img, axis=1)[0]
                groundtruth = lbl
                nbPixels += evalIoU.evaluatePairPytorch(prediction, groundtruth, confMatrix, perImageStats,
                                                        evalIoU.args)

    if doIouOrig:
        classScoreList = {}
        for label in evalIoU.args.evalLabels:
            labelName = evalIoU.trainId2label[label].name
            classScoreList[labelName] = evalIoU.getIouScoreForTrainLabel(label, confMatrix, evalIoU.args)

        iouAvgStr = evalIoU.getColorEntry(evalIoU.getScoreAverage(classScoreList, evalIoU.args),evalIoU.args) + \
                    "{avg:5.3f}".format( avg=evalIoU.getScoreAverage(classScoreList, evalIoU.args)) + \
                    evalIoU.args.nocol
        iouVal = float(evalIoU.getScoreAverage(classScoreList, evalIoU.args))
        print("EPOCH IoU on VAL set: ", iouAvgStr)
        print("")
        evalIoU.printClassScoresPytorchTrain(classScoreList, evalIoU.args)
        print("--------------------------------")
        print("Score Average : " + iouAvgStr )#+ "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    '''
    for i, (names, original, images, _) in enumerate(loader_val):
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            original = Variable(original.cuda())
        else:
            images = Variable(images)
        outputs = model(images)
        outputs = F.upsample_bilinear(outputs, (1024,2048))
        print("Overlay image ",i)
        overlay_images(names, original, outputs, i * batch_size, runs=runs)
        del outputs
        del images
    '''

    print("End Of Eval")
    model.train()
    return iouVal

def main():
    if train:
        train()
    else:
        model = duc_hdc.ResNetDUCHDC(num_classes)
        if resume:
            print("Resuming From ", resume_filename)
            checkpoint = torch.load(resume_filename)
            saved_state_dict = checkpoint['state_dict']
            model.load_state_dict(saved_state_dict)
            model.cuda()
            eval(model)

main()