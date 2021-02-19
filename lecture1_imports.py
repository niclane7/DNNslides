import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt
from time import time
from get_size import data_usage, num_params, num_mb, prod
from pthflops import count_ops
import hiddenlayer as hl

from functools import reduce
from torchvision.models import AlexNet
import torch.nn as nn
import matplotlib.pyplot as plt

def make_MNIST_loader(train=True, batch_size=128, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor()])
    dset = torchvision.datasets.MNIST(root='./data', train=train,
                                        download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=train, num_workers=num_workers)
    return loader

MNIST_trainloader = make_MNIST_loader()
MNIST_valloader = make_MNIST_loader(train=False)

def train(net, trainloader, num_epochs=2, save=False, prog_bar=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0

        if prog_bar:
            data_tqdm = tqdm.tqdm(enumerate(trainloader))
        else:
            data_tqdm = enumerate(trainloader)
        for i, (inputs, labels) in data_tqdm:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if prog_bar:
                data_tqdm.set_description(f'Epoch {epoch + 1}, iter {i + 1}, loss {running_loss/(i + 1):.3f}')
            if (i + 1) % 5000 == 0 and save:
                total = 0
                correct = 0
                with torch.no_grad(): # validate without computing gradients
                    for (val_imgs, val_labels) in MNIST_valloader:
                        outputs = our_custom_net(val_imgs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += val_labels.size(0)
                        correct += (predicted == val_labels).sum().item()
                    print(f'correct: {correct}, total: {total}, accuracy: {correct/total*100:.2f}%')
                # save our model parameters
                torch.save(our_custom_net.state_dict(), f'savedmodels/SimpleDLModel/epoch{epoch + 1}_iter{i + 1}.pth')

def gpu_train(net, trainloader, num_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0

        data_tqdm = tqdm.tqdm(enumerate(trainloader))
        for i, (inputs, labels) in data_tqdm:
            # zero the parameter gradients
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            data_tqdm.set_description(f'Epoch {epoch + 1}, iter {i + 1}, iter loss {running_loss/(i + 1):.3f}')

class BasicFCModel(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape((bs, -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.relu(x)

def estimate_training_for(nettype, num_epochs=1000):
    hidden_states = [128, 256, 512, 1024, 2048, 4096, 6144]
    times = []
    for num_hidden in hidden_states:
        net = nettype(num_hidden)
        custom_trainloader = make_MNIST_loader(batch_size=850, num_workers=2)
        start = time()
        train(net, custom_trainloader, num_epochs=1, prog_bar=False)
        total_time = time() - start
        times.append(total_time)
        print(f'Using {num_hidden} hidden nodes took {total_time:.2f} seconds,\
        training for {num_epochs} epochs would take ~{num_epochs * total_time}s')
    plt.plot(hidden_states, times)
    plt.title('Time taken vs number of hidden states')

class LeNet(nn.Module):
    def __init__(self, hidden=120):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(16*5*5, hidden)
        self.fc2   = nn.Linear(hidden, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

layers_of_interest = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)

def forward_hook(self, input, output):
    """ Stores input and ouptut shape for the given layer.
    Also stores the number of layer parameters (weights and
    biases) """

    self.input_shape = list(input[0].shape)
    self.output_shape = list(output.shape)
    self.num_parameters = 0
    if isinstance(self, (nn.Conv2d, nn.Linear)):
        self.num_parameters = self.weight.numel()
        if self.bias is not None:
            self.num_parameters += self.bias.numel()


def backward_hook(self, input_g, output_g):
    """ Stores incoming gradients' shape for the given layer.
    `input_g` will be a triplet including gradientw w.r.t to weights,
    gradients w.r.t. biases and, gradients w.r.t to inputs (that will
    be passed to the next layer as part of the chain rule. """

    self.gradient_shapes = []
    for g in input_g:
        if g is not None:  # gradients w.r.t to input for the first layer will be None
            self.gradient_shapes.append(list(g.shape))
        else:
            self.gradient_shapes.append([0])

    # for some reason the ordering of input_g is different for conv2d and linear layers
    # we want to keep the order as gradietns w.r.t input, weights, biases. We change the
    # order in Linear layers
    if isinstance(self, torch.nn.Linear):
        g_inputs = self.gradient_shapes[1]
        g_biases = self.gradient_shapes[0]
        g_weights = self.gradient_shapes[2]
        self.gradient_shapes = [g_inputs, g_weights, g_biases]

    # at the end the output gradients (which are the gradients passed)
    # form the layer_i+1
    self.gradient_shapes.append(list(output_g[0].shape))


def add_hooks(model):
    """ Add forward and backward hooks to Conv2d and Linear layer. """
    hooks = []
    for m in model.modules():
        if isinstance(m, layers_of_interest):
            hooks.append(m.register_forward_hook(forward_hook))
            hooks.append(m.register_backward_hook(backward_hook))


def prod(a_list):
    """ Multiply elements in list. """
    return reduce((lambda x, y: x * y), a_list)


def get_model_summary(model):
    print("Layer\tParameters\tInput shape\tOutput shape\t\tGradients shapes")
    print(f'{"=" * 110}')
    total_parameters = 0
    for m in model.modules():
        if isinstance(m, layers_of_interest):
            if isinstance(m, nn.AdaptiveAvgPool2d):
                name = "AvgPool"
            elif isinstance(m, nn.MaxPool2d):
                name = "MaxPool"
            else:
                name = m.__class__.__name__
            print(
                f"{name.ljust(10)}{str(m.num_parameters).ljust(10)}{str(m.input_shape).replace(' ', '').ljust(18)}{str(m.output_shape).replace(' ', '').ljust(18)}{str([gs for gs in m.gradient_shapes]).replace(' ', '')}")
            total_parameters += m.num_parameters
    print(f"Total number of parameters: {total_parameters} --> ~ {total_parameters / (1000 ** 2):.3} M")


def to_mb(num_parameters):
    return num_parameters * 4 / (1024 ** 2)


def get_working_set_info(model):
    layers = []
    for m in model.modules():
        if isinstance(m, layers_of_interest):
            name = "AvgPool2d" if isinstance(m, nn.AdaptiveAvgPool2d) else m.__class__.__name__
            inference_working_set = [m.num_parameters, prod(m.input_shape), prod(m.output_shape)]
            gradients_working_set = [prod(gs) for gs in m.gradient_shapes]
            layers.append([name, inference_working_set, gradients_working_set])
    return layers


def run_once(model, batch_size=1, input_shape=(3, 224, 224), num_classes=1000):
    dummy_input = torch.randn((batch_size, *input_shape))
    dummy_targets = torch.randint(0, num_classes - 1, size=(batch_size,))
    # inference
    out = model(dummy_input)

    # now backward
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, dummy_targets)
    loss.backward()



