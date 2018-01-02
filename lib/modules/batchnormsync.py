from queue import Queue

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from functions.batchnormp import BatchNormPFunction


class BatchNormSync(Module):

    sync = True
    checking_mode = False

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 device_ids=None):
        super(BatchNormSync, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.mean = torch.zeros(num_features)
        self.std = torch.ones(num_features)
        self.reset_parameters()
        self.cum_queue = Queue()
        self.broadcast_queue = Queue()
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.mean.zero_()
        self.std.fill_(1)
        if self.affine:
            if BatchNormSync.checking_mode:
                self.weight.data.fill_(1)
            else:
                self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        training = int(self.training)
        assert input.size(1) == self.num_features

        bn_func = BatchNormPFunction(
            self.running_mean, self.running_var, # self.mean, self.std,
            training, self.cum_queue, self.broadcast_queue, self.device_ids,
            BatchNormSync.sync, self.eps, self.momentum, self.affine)
        return bn_func(input, self.weight, self.bias)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))