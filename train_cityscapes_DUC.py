import tqdm
from PIL import Image
from graphviz import Digraph
from tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np

from dataset.CityscapesLoader_old import *
from models import deeplab_resnet_DUC
from utils import *

total_inference = 0

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
    imgs = (imgs * 255)
    imgs += np.array([104.00699, 116.66877, 122.67892])
    imgs = imgs[:,:,:,::-1]
    imgs = imgs.astype('uint8')
    return mask, imgs, out


def overlay_images(names, input, output, starting_index, runs="", convert_id = True):
    mask, imgs, out = tensor_to_imgs(input, output)
    for i in range(mask.shape[0]):
        temp = np.copy(out[i])
        if convert_id:
            trainId2Id = get_trainId2Id()
            for j in range(len(trainId2Id)):
                temp[out[i] == j] = trainId2Id[j]
                res = Image.fromarray(temp)
                res.save('/home/cattaneod/CITYSCAPES/results/'+names[i])
        else:
            m = Image.fromarray(mask[i])
            im = Image.fromarray(imgs[i])
            im.paste(m, box=None, mask=m)
            im.save('./CITYSCAPES/inference/'+str(runs)+str(starting_index + i)+".png")




def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())
    b.append(model.dense_upsample.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

num_classes = 19
image_shape = (256,512)
starting_epoch = 0
epochs = 20
batch_size = 1
l_rate = 2.5e-4
update_batches = 10

freeze_layers = False
resume = True
save = True
check_validation = False
overlay_during_training = True
poly_lr = True
TBWriter = True
TBUpdate = 100
num_workers=2

base_save_folder = "./CITYSCAPES/models/DUC"
base_data_folder = '/home/cattaneod/CITYSCAPES/'
#resume_filename = "./CITYSCAPES/models/checkpoint_17_0.335076503344.pth.tar"
resume_filename = base_save_folder + "/checkpoint_6_0.335163406079.pth.tar"
opt = "SGD"
reset_layer5 = False


data_augmentation = DataAugmentationTransform_old(
    flip_p = 0.5,
    rotation_range = 10,
    #zoom_range = (0.7, 1.4) ,
    #translation_range = (0.3, 0.3),
    #gamma_range = (0.5, 1.5),
    brightness_range = (-0.2, 0.2),
    saturation_range = (-0.2, 0.2),
    crop_size=(544,544),
    debug=False)

def train():

    loader_train = CityscapesLoader('/home/cattaneod/CITYSCAPES_crop/', split='train', is_transform=True, img_size=None, transforms=data_augmentation)
    trainloader = data.DataLoader(loader_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    loader_test = CityscapesLoader(base_data_folder, split='test', is_transform=True, img_size=None, transforms=data_augmentation)
    test_loader = data.DataLoader(loader_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    loader_val = CityscapesLoader(base_data_folder, split='val', is_transform=True, img_size=image_shape, return_original=True)
    valloader = data.DataLoader(loader_val, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    model = deeplab_resnet_DUC.Res_Deeplab_DUC(num_classes)

    if TBWriter:
        writer = SummaryWriter()

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
    else:
        print("Using CPU")

    model.train()

    if opt == "SGD":
        optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': l_rate }, {'params': get_10x_lr_params(model), 'lr': 10*l_rate} ], lr=l_rate, momentum=0.9, weight_decay=5e-4)
    elif opt =="Adam":
        optimizer = torch.optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': 0 * l_rate }, {'params': get_10x_lr_params(model), 'lr': 10*l_rate} ], lr=l_rate, weight_decay=5e-4)

    if resume:
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        if reset_layer5:
            for i in model.state_dict():
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                if i not in saved_state_dict or i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
        starting_epoch = checkpoint['epoch'] + 1
        if poly_lr:
            lr_ = poly_lr2(l_rate, len(trainloader) * starting_epoch, lr_decay_iter=1, max_iter=len(trainloader) * epochs)
            if lr_:
                if opt == "SGD":
                    optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_},
                                                 {'params': get_10x_lr_params(model), 'lr': 10 * lr_}],
                                                lr=lr_, momentum=0.9, weight_decay=5e-4)
                elif opt == "Adam":
                    optimizer = torch.optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': 0 * lr_},
                                                  {'params': get_10x_lr_params(model), 'lr': 10 * lr_}],
                                                 lr=lr_, weight_decay=5e-4)

    best_metric = 0
    old_file = ""
    train_acc = AverageMeter()
    train_IoU = AverageMeter()
    train_loss = AverageMeter()
    for epoch in range(starting_epoch, epochs):
        train_acc.reset()
        train_IoU.reset()
        train_loss.reset()
        train_cfmatrix = np.zeros((num_classes, num_classes))

        print("\nEpoch: ",epoch)

        if overlay_during_training and epoch % 1  == 0:
            for i in range(15):
                print("Overlaying image ",i)
                names, original_img, test_img, _ = loader_val[i]
                test_img = test_img.unsqueeze(0)
                original_img = original_img.unsqueeze(0)
                original_img = Variable(original_img.cuda())
                model.eval()
                test_pred = model(Variable(test_img.cuda(0), requires_grad=True))
                test_img = Variable(test_img.cuda(0), requires_grad=True)
                #if TBWriter and i==0:
                #    writer.add_graph(model, test_pred)
                test_pred = F.upsample_bilinear(test_pred, (1024, 2048))
                overlay_images(names, original_img, test_pred, epoch, str(i) + '_', convert_id=False)
                del test_pred
                del test_img

        model.train()
        optimizer.zero_grad()
        with tqdm.tqdm(trainloader, ncols=150) as t:
            lr_ = l_rate
            for i, (images, labels) in enumerate(t):
                if torch.cuda.is_available():
                    images = Variable(images.cuda(0))
                    labels = Variable(labels.cuda(0))
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                iter = len(trainloader) * epoch + i


                outputs = model(images)
                #g = make_dot(outputs)
                #g.save('./t.dot')

                loss = misc.cross_entropy2d(outputs, labels, ignore_index=255)
                loss = loss / update_batches

                loss.backward()

                t.set_description('Loss: %8.4f - LR = %f' % (update_batches * loss.data[0], lr_)        )

                train_loss.update(update_batches * loss.data[0])
                acc, IoU, cf_matrix =  accuracy_IoU(outputs,labels, np.array(range(num_classes)))
                if acc is not None:
                    train_acc.update(acc)
                    train_IoU.update(np.nanmean(IoU))
                    train_cfmatrix = train_cfmatrix + cf_matrix

                if i % update_batches == 0:
                    optimizer.step()
                    if poly_lr:
                        lr_ = poly_lr2(l_rate, iter, lr_decay_iter=1, max_iter=len(trainloader) * epochs)
                        if lr_:
                            t.set_description('Step: %8.4f - LR = %f' % (update_batches * loss.data[0], lr_))
                            if opt == "SGD":
                                optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_},
                                                             {'params': get_10x_lr_params(model), 'lr': 10 * lr_}],
                                                            lr=lr_, momentum=0.9, weight_decay=5e-4)
                            elif opt == "Adam":
                                optimizer = torch.optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': 0 * lr_},
                                                              {'params': get_10x_lr_params(model), 'lr': 10 * lr_}],
                                                             lr=lr_, weight_decay=5e-4)

                    #print("%8.2f %%  ->  Loss: %8.6f " % (i / len(trainloader) * 100, loss.data[0]), end='\r')
                    optimizer.zero_grad()

                if i > 0 and i % TBUpdate == 0 and TBWriter:
                    writer.add_scalar('Train Accuracy', train_acc.avg, iter)
                    writer.add_scalar('Train IoU', train_IoU.avg, iter)
                    writer.add_scalar('Train Loss', train_loss.avg, iter)

                del outputs
                del loss
                del images
                del labels

                t.update(1)


                rows = train_cfmatrix.sum(axis=1)
        cols = train_cfmatrix.sum(axis=0)
        IoU = np.ndarray(train_cfmatrix.shape[0])
        for i in range(train_cfmatrix.shape[0]):
            if rows[i] + cols[i] > 0.:
                IoU[i] = train_cfmatrix[i][i] / (rows[i] + cols[i] - train_cfmatrix[i][i])
            else:
                IoU[i] = np.nan
        print("\nTrain Accuracy: ", train_acc.avg)
        print("Train Loss: ", train_loss.avg)
        print("Micro IoU: ", train_IoU.avg, "\n")
        print("Macro IoU: ", np.nanmean(IoU), "\n")

        if check_validation:
            #VALIDATION!!!
            val_acc = AverageMeter()
            val_IoU = AverageMeter()
            val_loss = AverageMeter()
            val_cfmatrix = np.zeros((num_classes, num_classes))
            model.eval()
            for i, (images, labels) in enumerate(valloader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda(0))
                    labels = Variable(labels.cuda(0))
                else:
                    images = Variable(images)
                    labels = Variable(labels)
                iter = len(trainloader) * epoch + i
                #poly_lr_scheduler(optimizer, l_rate, iter)

                outputs = model(images)

                loss = cross_entropy2d(outputs, labels, ignore_index=255)

                val_loss.update(loss.data[0])
                acc, IoU, cf_matrix = accuracy_IoU(outputs,labels, np.array(range(num_classes)))
                if acc is not None:
                    val_acc.update(acc)
                    val_IoU .update(np.nanmean(IoU))
                    val_cfmatrix = val_cfmatrix + cf_matrix

                del outputs
                del loss
                del images
                del labels
            print("\nVal Accuracy: ", val_acc.avg)
            print("Val Loss: ", val_loss.avg)
            print("Val IoU: ", val_IoU.avg, "\n")
            if TBWriter:
                writer.add_scalar('Val Accuracy', val_acc.avg, epoch)
                writer.add_scalar('Val IoU', val_IoU.avg, epoch)
                writer.add_scalar('Val Loss', val_loss.avg, epoch)

        save_metric = train_IoU.avg
        if check_validation:
            save_metric = val_IoU.avg

        if best_metric < save_metric:
            best_metric = save_metric
            print("New Best IoU!")
            if save:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                 base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar")
                print("Model Saves As " + base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar")
                if os.path.isfile(old_file):
                    os.remove(old_file)
                old_file = base_save_folder + "/checkpoint_" + str(epoch) + "_" + str(save_metric) + ".pth.tar"

        print("Best IoU So Far: ", best_metric)

    if TBWriter:
        writer.close()
    print("End Of Training")


from datetime import datetime

def eval():
    loader = CityscapesLoader(base_data_folder, split='val', is_transform=True, img_size=image_shape, return_original=True)
    val_loader = data.DataLoader(loader, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    model = deeplab_resnet_DUC.Res_Deeplab_DUC(num_classes)

    if resume:
        print("Loading from: ", resume_filename)
        checkpoint = torch.load(resume_filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Using not trained model, are u sure??")

    if torch.cuda.is_available():
        print("Using GPU")
        model.cuda(0)
    else:
        print("Using CPU")

    model.eval()
    runs = datetime.now().strftime('%b%d_%H-%M-%S')+"/"
    os.makedirs('./CITYSCAPES/inference/'+str(runs))
    for i, (names, original, images, _) in enumerate(val_loader):
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

    print("End Of Eval")

train()
