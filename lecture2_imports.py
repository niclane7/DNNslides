import torch
from torch import nn
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.nn import functional as F
from torch import optim
from torch_geometric import nn as pyg_nn
from torch_geometric.datasets import QM9
import networkx as nx
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
from PIL import Image
from dlgdrive import download_file_from_google_drive

from srgan.utils import *
import models
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

transform = transforms.Compose(
    [transforms.ToTensor()])

CIFAR10_trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
CIFAR10_trainloader = torch.utils.data.DataLoader(CIFAR10_trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

def onedimsz(xysz, kernel_size, stride):
    return int((xysz - kernel_size)/stride + 1)

def output_im(image, kernel, stride):
    imx, imy, _ = image.shape
    kernel_size, _, _ = kernel.shape
    outszx = onedimsz(imx, kernel_size, stride)
    outszy = onedimsz(imy, kernel_size, stride)
    return np.zeros((outszx, outszy))

def num_strides(image, kernel, stride):
    imx, imy, _ = image.shape
    c, _, _ = kernel.shape
    if (imx - c) % stride != 0 or (imy - c) % stride != 0:
        raise ArithmeticError(f'Image shape ({imx}, {imy}) does not allow striding with stride {stride} evenly')
    return (imx - c) // stride + 1, (imy - c) // stride + 1

def pad(image, padding):
    imx, imy, imz = image.shape
    new_image = np.zeros((imx + 2*padding, imy + 2*padding, imz))
    new_image[padding:imx+padding, padding:imy+padding] = image
    return new_image

def convolve_to(num, image, kernel, stride, padding):
    image = pad(image, padding)
    output = output_im(image, kernel, stride)
    num_stridesx, num_stridesy = num_strides(image, kernel, stride)
    c, _, _ = kernel.shape
    for i in range(num):
        x, y = i//num_stridesx, i%num_stridesy
        startx, starty = x * stride, y * stride
        output[x, y] = np.sum(image[startx:startx+c, starty:starty+c, :] * kernel)
    return shiftrescale(output) * 255

def showuntil(num, im, data):
    imx, imy = data.shape
    output = np.zeros_like(data)
    for i in range(num):
        x, y = i // imy, i % imy
        output[x, y] = data[x, y]
    im.set_data(output)
    return output

def shiftrescale(image):
    shift = image + np.abs(np.min(image))
    rescaled = shift / np.max(shift)
    return rescaled

class socfb(InMemoryDataset):
    # @inproceedings{nr,
    #      title={The Network Data Repository with Interactive Graph Analytics and Visualization},
    #      author={Ryan A. Rossi and Nesreen K. Ahmed},
    #      booktitle={AAAI},
    #      url={http://networkrepository.com},
    #      year={2015}
    # }
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['readme.html', f'{self.name}.mtx']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        from urllib.request import urlopen
        a = urlopen(f'http://nrvis.com/download/data/socfb/{self.name}.zip')
        if a.status == 200:
            with open(f'{self.root}/raw/{self.name}.zip', 'wb') as f:
                f.write(a.read())
        else:
            raise Exception(f'Could not download {self.name}')
        from zipfile import ZipFile
        with ZipFile(f'{self.root}/raw/{self.name}.zip') as f:
            f.extractall(path=f'{self.root}/raw')

    def process(self):
        # Read data into huge `Data` list.
        with open(f'{self.root}/raw/{self.raw_file_names[1]}') as f:
            lines = f.readlines()[1:] # First line is a comment I think
            num_nodes, a, num_edge = lines[0].split()
            lines = lines[1:]
            assert(len(lines) == int(num_edge))
            edges = [line.split() for line in lines]
            edges = [[int(x[0]), int(x[1])] for x in edges]
            edges = torch.tensor(edges, dtype=torch.long).T
        data_list = [Data(edge_index=edges)]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def update_image(num, fig, im, data):
    showim = showuntil(num, im, data)
    setfigval(fig, showim)
    return im

def setfigval(fig, vals):
    xsteps, ysteps = vals.shape
    if len(fig.texts) == xsteps * ysteps:
        for num, txt in enumerate(fig.texts):
            i, j = num // xsteps, num % ysteps
            txt.set_text(f'{vals[i, j]:.2f}')
    else:
        for i in range(xsteps):
            for j in range(ysteps):
                txt = plt.text(j, i, f'{vals[i, j]:.1f}', ha='center', va='center', color='white')
                fig.texts.append(txt)

def show_car(CIFAR10_trainset):
    plt.figure()
    data, label = CIFAR10_trainset[4]
    datashow = data.transpose(0, 1).transpose(1, 2)
    plt.imshow(datashow)
    plt.title('Image')
    return data

def show_random_kernel():
    plt.figure()
    kernel_size = 8
    stride = 8
    padding = 0
    conv = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=stride, padding=padding)
    kernel = conv.weight.squeeze().transpose(0, 1).transpose(1, 2).detach()
    plt.imshow(shiftrescale(kernel.numpy()))
    plt.title('Kernel')
    return conv

def show_conv_anim(data, conv):
    fig = plt.figure()

    stride = 8
    padding = 0
    datashow = data.transpose(0, 1).transpose(1, 2)
    padded = pad(datashow, padding)
    kernel = conv.weight.squeeze().transpose(0, 1).transpose(1, 2).detach()
    xsteps, ysteps = num_strides(padded, kernel, stride)
    num_steps = xsteps * ysteps + 1
    data = data.unsqueeze(0)
    final_fm = conv(data)
    final_fm = final_fm.squeeze().squeeze().detach().numpy()
    fm_min, fm_max = np.min(final_fm), np.max(final_fm)
    im = plt.imshow(np.zeros_like(final_fm), vmin=fm_min, vmax=fm_max) # vmin and vmax required so the image isn't blank
    # setfigval(fig1, final_fm)
    # print(update_image(num_steps-1, fig1, im, final_fm).get_array())
    plt.title(f'Output of convolution (feature map)')
    return fig, num_steps, im, final_fm

def update_image_no_annotation(num, im, data):
    showim = showuntil(num, im, data)
    return im

def show_cute_fox():
    plt.figure()
    data = torch.tensor(np.array(Image.open('fox.jpeg')))
    x, y, _ = data.shape
    data = data[:min(x, y):4, :min(x, y):4, :]
    datashow = data.clone()
    data = data.transpose(1, 2).transpose(0, 1).unsqueeze(0).type(torch.DoubleTensor)

    plt.imshow(datashow)
    plt.title('Image')
    return data

def show_edge_detector(kernel):
    plt.figure()
    # Change the kernel and see what happens!
    stride = 1
    padding = 1

    conv = nn.Conv2d(3, 1, kernel_size=kernel.shape, stride=stride, padding=padding)
    kernel = np.stack((kernel,)*3, axis=0)
    kernelshow = shiftrescale(kernel).T
    conv.weight = nn.Parameter(torch.tensor(kernel).unsqueeze(0).type(torch.DoubleTensor), requires_grad=False)
    plt.imshow(kernelshow)
    plt.title('Kernel')
    return conv

def show_edge_detection(data, conv):
    fig = plt.figure()

    stride = 1
    padding = 1
    datashow = data.squeeze().transpose(0, 1).transpose(1, 2)
    padded = pad(datashow, padding)
    kernel = conv.weight.squeeze().transpose(0, 1).transpose(1, 2).detach()
    xsteps, ysteps = num_strides(padded, kernel, stride)
    num_steps = xsteps * ysteps + 1
    final_fm = conv(data)
    final_fm = final_fm.squeeze().squeeze().detach().numpy()
    fm_min, fm_max = np.min(final_fm), np.max(final_fm)
    im = plt.imshow(np.zeros_like(datashow), cmap='gray', vmin=fm_min, vmax=fm_max) # vmin and vmax required so the image isn't blank
    plt.title('Output of convolution (feature map)')
    return fig, num_steps, im, final_fm

def update_row(num, ims, imlist):
    for im, d in zip(ims, imlist[num*4:(num+1)*4, :, :, :]):
        im.set_data(d.detach().numpy())
    return ims

def get_mobilenet_convbnrelu():
    mob = mobilenet_v2(pretrained=True)
    modules = list(mob.modules())
    convbnrelu = modules[2]
    return convbnrelu

def show_conv_weights(convlayer):
    num_steps = 8
    fig, axes = plt.subplots(1, 4, figsize=(9, 4))
    fig.suptitle('Convolutional layer weights')
    ims = [ax.imshow(shiftrescale(kernel.transpose(1, 2).transpose(0, 1).detach().numpy())) for ax, kernel in zip(axes, convlayer.weight[:4, :, :, :])]
    return fig, num_steps, ims

def get_224px_fox():
    data = Image.open('fox.jpeg')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    data = transform(data)
    return data

def show_224px_fox():
    fig = plt.figure()
    fig.suptitle('Still using our cute fox but in 224x224')
    data = get_224px_fox()
    plt.imshow(data.transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
    return data

def update_row_output(num, ims, imlist):
    for im, d in zip(ims, imlist[num*4:(num+1)*4, :, :]):
        im.set_data(d.detach().numpy())
    return ims

def show_layer_output(data, layer):
    num_steps = 8
    fig, axes = plt.subplots(1, 4, figsize=(9, 4))
    fig.suptitle('Convolutional layer feature map outputs')
    output = layer(data.unsqueeze(0)).squeeze()
    ims = [ax.imshow(fm.detach().numpy()) for ax, fm in zip(axes, output[:4, :, :])]
    return fig, num_steps, ims, output

def show_maxpool(data, maxpool):
    plt.figure()
    maxpooloutput = maxpool(data)
    plt.title('Maxpool output')
    plt.imshow(shiftrescale(maxpooloutput.transpose(0, 1).transpose(1, 2).numpy()))

def show_DW_conv(data, conv3x3, dwconv3x3, dwconv1x1):
    dw_3x3_out = torch.stack([dwconv3x3(data[i,:,:].unsqueeze(0).unsqueeze(0)).squeeze() for i in range(3)])
    dw_1x1_out = dwconv1x1(dw_3x3_out.unsqueeze(0)).squeeze()
    conv_3x3_out = conv3x3(data.unsqueeze(0)).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(9,6))
    axes[0].set_title('DW separable convolution')
    axes[0].imshow(shiftrescale(dw_1x1_out.detach().numpy()))
    axes[1].set_title('Normal 3x3 convolution')
    axes[1].imshow(shiftrescale(conv_3x3_out.detach().numpy()))

def get_srgan_weights(srgan_checkpoint):
    if not os.path.isfile(srgan_checkpoint):
        download_file_from_google_drive('1_PJ1Uimbr0xrPjE8U3Q_bG7XycGgsbVo', srgan_checkpoint)

def tensor2im(tensor):
    return tensor.transpose(0, 1).transpose(1, 2).detach().numpy()

def show_srgan_comparison(srgan_checkpoint):
    srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator']
    srgan_generator.eval()
    data = Image.open('fox.jpeg')
    transform = ImageTransforms('test', crop_size=0, scaling_factor=4, lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    sr_data_lr, sr_data_hr = transform(data)
    sr_output = srgan_generator(sr_data_lr.unsqueeze(0)).squeeze()
    sr_output_im = (sr_output + 1.) / 2. 
    sr_data_lr_im = sr_data_lr * imagenet_std + imagenet_mean
    sr_data_hr_im = (sr_data_hr + 1.) / 2. 
    images_in_order = ('Low', tensor2im(sr_data_lr_im)), ('Original', tensor2im(sr_data_hr_im)), ('Output', tensor2im(sr_output_im))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for (name, im), ax in zip(images_in_order, axes):
        ax.set_title(name)
        ax.imshow(im)


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        if self.task == 'node':
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim+29, hidden_dim), nn.Dropout(0.25),
                nn.Linear(hidden_dim, output_dim))
        elif self.task == 'link':
            pass
        elif self.task == 'graph':
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
                nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node' or self.task == 'link':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'node':
            oh = torch.zeros(data.batch.shape[0], 29).to(x.device)
            print(data.mask)
            oh.scatter_(1, data.mask.unsqueeze(0), 1)
            x = torch.cat((x, oh), dim=1)
        elif self.task == 'link':
            pass
        elif self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        if self.task == 'node':
            return F.nll_loss(pred, label)
        elif self.task == 'graph':
            return F.mse_loss(pred, label)

def show_graph_regression():
    qm9 = QM9(root='data')
    test_loader = DataLoader(qm9[int(1000* 0.8):1000], batch_size=1, shuffle=True)
    model = GNNStack(max(qm9.num_node_features, 1), 32, qm9.num_classes, task='graph')
    model.load_state_dict(torch.load('graphnn/savegraphmodel_32hid.pth'))
    example = next(iter(test_loader))
    emb, pred = model(example)
    fig, axes = plt.subplots(2, 1, figsize=(10,4))
    fig.suptitle('Graph property prediction')
    axes[0].imshow(pred.detach().numpy())
    axes[0].set_title('Prediction')
    axes[1].imshow(example.y.detach().numpy())
    axes[1].set_title('Ground truth')
