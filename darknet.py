from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from utils import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # cache outputs for route layer
        write = 0     # indicate whether we have met our first detection
        for i, module in enumerate(modules):
            module_type = (module['type'])
            # for conv and upsample layer, just forward through this module
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            # deal with route layer, this is where we actually route
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            # deal with shortcut(residual)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i + from_]

            # deal with yolo layers
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                # cuda must be checked, otherwise you'll get a error calling func predict_transform()
                if CUDA:
                    x = x.cuda()
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        """
        loading weights from pretrained weightsfile, in this case ,we use
        coco 80 categories weightsfile
        :param weightfile: pretrained weightsfile
        :return: None, load weights to our model
        """
        fp = open(weightfile, 'rb')

        # The first 160 bytes of the weights file store 5 int32 values which
        # constitute the header of the file
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # store the rest weight file as a ndarray
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            # load weights when we meet conv layers since weights only exist at conv layer
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # get num of weights of bn layer
                    num_bn_biases = bn.bias.numel()
                    # load weights to torch tensor
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights so that dims match model
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                # load weights for convolutional layers
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def parse_cfg(cfgfile):
    """
    function parse cfg file
    :param cfgfile: configurations file which describe it's architecture
    :return: a list of blocks, each block describe a block in the neural network
    to be built.
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, val = line.split('=')
            block[key.rstrip()] = val.lstrip()
    # add last block since we won't met '[' again
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    :param blocks:  blocks returned by func parser_cfg()
    :return: module list and network information described by blocks
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # conv module
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(idx), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(idx), bn)

            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(idx), activ)

        # upsample module
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module('upsample_{0}'.format(idx), upsample)

        # route module, cache this position(index)
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            route = EmptyLayer()
            module.add_module('route_{0}'.format(idx), route)
            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                filters = output_filters[idx + start]

        # shortcut module, like ResNet
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(idx), shortcut)

        # yolo module, detection module
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(idx), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


# # test
# blocks = parse_cfg('cfg/yolov3.cfg')
# print(create_modules(blocks))
#
# model = Darknet("cfg/yolov3.cfg")
# model.load_weights('yolov3.weights')
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print(pred)
# print(pred.size())
