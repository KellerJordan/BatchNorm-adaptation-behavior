import os
from tqdm import tqdm
import torch
from torch import nn

import airbench
from airbench import CifarLoader, evaluate
test_loader = CifarLoader('/tmp/cifar10', train=False, batch_size=1000)


#############################################
#           Network Reset Utils             #
#############################################

def reset_bn(net, loader):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()
    with torch.no_grad():
        net.train()
        for inputs, _ in loader:
            net(inputs)

def fuse_conv_bn(conv, bn):
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    ).half().cuda()

    # setting weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused.weight.data[:] = (w_conv * gamma.reshape(-1, 1, 1, 1))

    # setting bias
    b_conv = conv.bias if conv.bias is not None else torch.zeros_like(bn.bias)
    beta = bn.bias + gamma * (-bn.running_mean + b_conv)
    fused.bias.data[:] = beta
    
    return fused

class ResetConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels).to(conv.weight.device)
        self.rescale = False
        
    def set_stats(self, goal_mean, goal_var):
        self.bn.bias.data = goal_mean
        goal_std = (goal_var + 1e-5).sqrt()
        self.bn.weight.data = goal_std
        
    def forward(self, x):
        x = self.conv(x)
        if self.rescale:
            x = self.bn(x)
        else:
            self.bn(x)
        return x
    
    def fuse(self):
        return fuse_conv_bn(self.conv, self.bn)

def make_tracked_net(net):
    net1 = make_net_normfree()
    net1.load_state_dict(net.state_dict())
    for block in net1[2:5]:
        block.conv1 = ResetConv(block.conv1)
        block.conv2 = ResetConv(block.conv2)
    return net1

def fuse_tracked_net(net):
    for block in net[2:5]:
        block.conv1 = block.conv1.fuse()
        block.conv2 = block.conv2.fuse()
    return net

def pseudo_reset(net, src_loader, tgt_loader):
    wrap0 = make_tracked_net(net)
    wrap1 = make_tracked_net(net)
    reset_bn(wrap0, src_loader)
    
    for m0, m1 in zip(wrap0.modules(), wrap1.modules()):
        if isinstance(m0, ResetConv):
            m1.set_stats(m0.bn.running_mean, m0.bn.running_var)
            m1.rescale = True

    # reset the tracked mean/var and fuse rescalings back into conv layers 
    reset_bn(wrap1, tgt_loader)
    return fuse_tracked_net(wrap1)

#############################################
#       NormFree Network Definition         #
#############################################

hyp = { 
    'net': {
        'widths': {
            'block1': 64, 
            'block2': 256,
            'block3': 256,
        },  
        'scaling_factor': 1/9,
    },  
}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
        w[w.size(1):] *= 3**0.5

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = Conv(channels_out, channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        return x

def make_net_normfree():
    widths = hyp['net']['widths']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1']),
        ConvGroup(widths['block1'], widths['block2']),
        ConvGroup(widths['block2'], widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    return net


#############################################
#                Make Figure                #
#############################################

if __name__ == '__main__':
    corruptions = os.listdir('corruption_data')

    net_normfree = make_net_normfree()
    normfree_path = 'sd_normfree.pt'
    net_normfree.load_state_dict(torch.load(normfree_path))
    print(evaluate(net_normfree, test_loader))

    batchnorm_path = 'sd_batchnorm.pt'
    net_batchnorm = airbench.make_net94()
    net_batchnorm.load_state_dict(torch.load(batchnorm_path))
    print(evaluate(net_batchnorm, test_loader))


    src_loader = CifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4))
    results = []
    corruptions1 = ['clean test set']+corruptions
    for k in tqdm(corruptions1):

        if 'clean' not in k:
            tgt_loader = CifarLoader('corruption_data/%s' % k, train=False)
        else:
            tgt_loader = CifarLoader('/tmp/cifar10', train=False)
        
        net_normfree.load_state_dict(torch.load(normfree_path))
        results.append(evaluate(net_normfree, tgt_loader))
        net_normfree_reset = pseudo_reset(net_normfree, src_loader, tgt_loader)
        results.append(evaluate(net_normfree_reset, tgt_loader))

        net_batchnorm.load_state_dict(torch.load(batchnorm_path))
        results.append(evaluate(net_batchnorm, tgt_loader))
        reset_bn(net_batchnorm, tgt_loader)
        results.append(evaluate(net_batchnorm, tgt_loader))

    torch.save(results, 'results.pt')

