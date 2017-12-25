import tqdm
from PIL import Image
from graphviz import Digraph
from tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset.CityscapesLoader_old import *
from models import *

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
    mask = out
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    mask = np.dot(mask, np.array([[0, 1, 1, 1]]))
    id2color = get_id2color()
    for i in range(num_classes):
        mask[out==i] = np.concatenate((id2color[i], [127]))
    mask = mask.astype(np.uint8)
    out = out * 255
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
    imgs = imgs[:, :, :, ::-1]
    imgs = imgs.astype('uint8')
    return mask, imgs


def overlay_images(input, output, starting_index, runs=""):
    mask, imgs = tensor_to_imgs(input, output)
    for i in range(mask.shape[0]):
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

num_classes = 19
image_shape = (256, 864)
starting_epoch = 0
epochs = 30
batch_size = 4
l_rate = 2.5e-4

freeze_layers = False
resume = True
save = True
check_validation = True
overlay_during_training = True
poly_lr = False

base_save_folder = "./CITYSCAPES/saved"
base_data_folder = '/home/cattaneod/CITYSCAPES/'
resume_filename = base_save_folder + "/checkpoint_2_0.268364420871.pth.tar"
opt = "Adam"

def train():
    data_augmentation = DataAugmentationTransform_old(translation_range=(0.0, 0.15),
                                                      rotation_range=10,
                                                      zoom_range = (0.8, 1.0),
                                                      flip_p = 0.5,
                                                      brightness_range = (-0.2, 0.2),
                                                      gamma_range = (0.5, 1.5),
                                                      saturation_range=(-0.3, 0.3))
    loader_train = CityscapesLoader(base_data_folder, split='train', is_transform=True, img_size=image_shape, transforms=None)
    trainloader = data.DataLoader(loader_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    if overlay_during_training:
        loader_test = CityscapesLoader(base_data_folder, split='test', is_transform=True, img_size=image_shape)
        test_loader = data.DataLoader(loader_test, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    if check_validation:
        loader_val = CityscapesLoader(base_data_folder, split='val', is_transform=True, img_size=image_shape)
        valloader = data.DataLoader(loader_val, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    model = get_model('fcn1s',num_classes)

    writer = SummaryWriter()

    if resume:
        print("Resuming From ",resume_filename)
        checkpoint = torch.load(resume_filename)
        model.load_state_dict(checkpoint['state_dict'])
        #starting_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])

    for param in model.parameters():
        param.requires_grad = True

    if freeze_layers:
        print("Freezing VGG layers")
        for param in model.conv_block1.parameters():
            param.requires_grad = False
        for param in model.conv_block2.parameters():
            param.requires_grad = False
        for param in model.conv_block3.parameters():
            param.requires_grad = False
        for param in model.conv_block4.parameters():
            param.requires_grad = False
        for param in model.conv_block5.parameters():
            param.requires_grad = False

    if torch.cuda.is_available():
        print("Using GPU")
        model.cuda(0)
    else:
        print("Using CPU")

    model.train()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if opt == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=l_rate, momentum=0.9, weight_decay=5e-4)
    elif opt =="Adam":
        optimizer = torch.optim.Adam(parameters, lr=l_rate, weight_decay=5e-4)

    best_metric = 0
    old_file = ""
    for epoch in range(starting_epoch, epochs):
        train_acc = 0
        train_IoU = 0
        train_loss = 0
        train_count = 0

        print("\nEpoch: ",epoch)

        if overlay_during_training and epoch % 5  == 0:
            test_img = loader_test[67]
            test_img = test_img.unsqueeze(0)
            model.eval()
            test_pred = model(Variable(test_img.cuda(0), requires_grad=True))
            test_img = Variable(test_img.cuda(0), requires_grad=True)
            overlay_images(test_img, test_pred, epoch, '67_')
            writer.add_graph(model, test_pred)
            del test_pred
            del test_img

            test_img = loader_test[88]
            test_img = test_img.unsqueeze(0)
            test_pred = model(Variable(test_img.cuda(0), requires_grad=True))
            test_img = Variable(test_img.cuda(0), requires_grad=True)
            overlay_images(test_img, test_pred, epoch, '88_')
            del test_pred
            del test_img

            test_img = loader_test[175]
            test_img = test_img.unsqueeze(0)
            test_pred = model(Variable(test_img.cuda(0), requires_grad=True))
            test_img = Variable(test_img.cuda(0), requires_grad=True)
            overlay_images(test_img, test_pred, epoch, '175_')
            del test_pred
            del test_img

        model.train()
        with tqdm.tqdm(trainloader, ncols=100) as t:
            for i, (images, labels) in enumerate(t):
                if torch.cuda.is_available():
                    images = Variable(images.cuda(0))
                    labels = Variable(labels.cuda(0))
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                iter = len(trainloader) * epoch + i
                if poly_lr:
                    poly_lr_scheduler(optimizer, l_rate, iter, lr_decay_iter=10)

                optimizer.zero_grad()
                outputs = model(images)

                loss = cross_entropy2d(outputs, labels, ignore_index=255)

                loss.backward()
                optimizer.step()

                #print("%8.2f %%  ->  Loss: %8.6f " % (i / len(trainloader) * 100, loss.data[0]), end='\r')
                t.set_description('Loss: %8.6f' % loss.data[0])
                t.update(1)

                train_loss = train_loss + loss.data[0]
                acc, IoU = accuracy_IoU(outputs,labels, np.array(range(num_classes)))
                train_acc = train_acc + acc
                train_IoU = train_IoU + IoU.mean()
                train_count = train_count + 1

                del outputs
                del loss
                del images
                del labels


        train_acc = train_acc / train_count
        train_IoU = train_IoU / train_count
        train_loss = train_loss / train_count
        print("\nTrain Accuracy: ", train_acc)
        print("Train Loss: ", train_loss)
        print("Train IoU: ", train_IoU, "\n")
        writer.add_scalar('Train Accuracy', train_acc, epoch)
        writer.add_scalar('Train IoU', train_IoU, epoch)
        writer.add_scalar('Train Los', train_loss, epoch)

        if check_validation:
            #VALIDATION!!!
            val_acc = 0
            val_IoU = 0
            val_loss = 0
            val_count = 0
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

                val_loss = val_loss + loss.data[0]
                acc, IoU = accuracy_IoU(outputs,labels, np.array(range(num_classes)))
                val_acc = val_acc + acc
                val_IoU = val_IoU + IoU.mean()
                val_count = val_count + 1

                del outputs
                del loss
                del images
                del labels
            val_acc = val_acc / val_count
            val_IoU = val_IoU / val_count
            val_loss = val_loss / val_count
            print("\nVal Accuracy: ", val_acc)
            print("Val Loss: ", val_loss)
            print("Val IoU: ", val_IoU, "\n")
            writer.add_scalar('Val Accuracy', val_acc, epoch)
            writer.add_scalar('Val IoU', val_IoU, epoch)
            writer.add_scalar('Val Loss', val_loss, epoch)

        save_metric = val_IoU
        if check_validation:
            save_metric = val_IoU

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

    writer.close()
    print("End Of Training")


from datetime import datetime

def eval():
    loader = CityscapesLoader(base_data_folder, split='test', is_transform=True, img_size=image_shape)
    val_loader = data.DataLoader(loader, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    model = get_model('fcn1s',num_classes)

    if resume:
        print("Resuming From ",resume_filename)
        checkpoint = torch.load(resume_filename)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
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
    for i, (images) in enumerate(val_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
        else:
            images = Variable(images)
        outputs = model(images)
        print("Overlay image ",i)
        overlay_images(images, outputs, i * batch_size, runs=runs)
        del outputs
        del images

    print("End Of Eval")