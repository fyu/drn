import pdb

import numpy as np

import torch
from torch.autograd import Function
from dense import batch_norm

from queue import Queue
from threading import Condition

cum_queue = Queue()
broadcast_queue = Queue()
broadcast_cv = Condition()


class BatchNormPFunction(Function):
    def __init__(self, running_mean, running_var, training,
                 cum_queue, broadcast_queue, device_ids, sync,
                 eps=1e-5, momentum=0.1, affine=True):
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        self.mean = None
        self.var = None
        self.training = training
        self.cum_queue = cum_queue
        self.broadcast_queue = broadcast_queue
        self.device_ids = device_ids
        self.sync = sync

    def forward(self, input, weight, bias):
        output = input.new()
        self.save_for_backward(input, weight, bias)

        # input_t = input.transpose(0, 1).double()
        # input_size = input_t.size()
        batch_size = int(input.size(0))
        # input_t.resize_(int(input_size[0]), int(np.prod(input_size[1:])))
        # self.mean = input_t.mean(dim=1)

        device_ids = self.device_ids
        # print('device', input.get_device(), flush=True)
        if input.is_cuda:
            # self.mean.copy_(torch.from_numpy(
            #     self.cum_mean(input.get_device(),
            #                   self.mean.cpu().numpy(),
            #                   batch_size)))
            # var = input_t - torch.unsqueeze(self.mean, 1)
            # var *= var
            # var = var.mean(dim=1)
            # total_var = self.cum_mean(
            #     input.get_device(), var.cpu().numpy(), batch_size)
            # self.std = input_t.new().resize_as_(self.mean). \
            #     copy_(torch.from_numpy(total_var)).sqrt()

            mean_cuda = input.new().resize_(input.size(1))
            var_cuda = input.new().resize_(input.size(1))
            batch_norm.BatchNormalizationP_mean_cuda(input, mean_cuda)

            if len(device_ids) > 1 and self.sync and self.training:
                mean_cuda.copy_(torch.from_numpy(self.cum_mean(
                    input.get_device(), mean_cuda.cpu().numpy(), batch_size)))
            batch_norm.BatchNormalizationP_var_cuda(input, mean_cuda, var_cuda)
            if len(device_ids) > 1 and self.sync and self.training:
                var_cuda.copy_(torch.from_numpy(self.cum_mean(
                    input.get_device(), var_cuda.cpu().numpy(), batch_size)))
        else:
            # self.std = input_t.std(dim=1, unbiased=False)
            batch_norm.BatchNormalizationP_var_cuda(input, mean_cuda, var_cuda)
        self.mean = mean_cuda
        self.var = var_cuda

        if not input.is_cuda:
            self.std = input_t.std(dim=1, unbiased=False)
            batch_norm.BatchNormalizationP_forward(
                input, output, weight, bias,
                self.running_mean, self.running_var, self.mean, self.std,
                self.training, self.momentum, self.eps)
        else:
            batch_norm.BatchNormalizationP_forward_cuda(
                input, output, weight, bias,
                self.running_mean, self.running_var, self.mean, self.var,
                self.training, self.momentum, self.eps)
        return output

    def cum_mean(self, this_device, this_mean, batch_size):
        cum_queue.put((batch_size, this_mean))
        total_mean = np.zeros(this_mean.shape, dtype=np.float64)
        total_batch_size = 0
        if this_device == self.device_ids[0]:
            for _ in self.device_ids:
                item = cum_queue.get()
                total_batch_size += item[0]
                total_mean += item[0] * item[1]
                cum_queue.task_done()
            total_mean /= total_batch_size
            broadcast_cv.acquire()
            for _ in range(len(self.device_ids) - 1):
                broadcast_queue.put(total_mean)
            broadcast_cv.notify_all()
            broadcast_cv.release()
        else:
            broadcast_cv.acquire()
            if broadcast_queue.qsize() == 0:
                broadcast_cv.wait()
            total_mean = broadcast_queue.get()
            broadcast_queue.task_done()
            broadcast_cv.release()
        # assert cum_queue.empty()
        broadcast_queue.join()
        return total_mean

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors
        grad_input = grad_output.new().resize_as_(input)
        grad_weight = grad_output.new().resize_as_(weight).zero_()
        grad_bias = grad_output.new().resize_as_(bias).zero_()
        if not grad_output.is_cuda:
            batch_norm.BatchNormalizationP_backward(
                input, grad_output, grad_input, grad_weight, grad_bias,
                weight, self.running_mean, self.running_var, self.mean,
                self.std, self.training, 1, self.eps)
        else:
            # grad_output_t = grad_output.transpose(0, 1).double()
            # batch_size = int(grad_output.size(0))
            # grad_output_t.resize_(int(grad_output_t.size(0)),
            #                       int(np.prod(grad_output_t.size()[1:])))
            # grad_output_mean = grad_output_t.mean(dim=1)
            # device_ids = self.device_ids
            # if len(device_ids) > 1 and self.sync:
            #     grad_output_mean.copy_(torch.from_numpy(
            #         self.cum_mean(grad_output.get_device(),
            #                       grad_output_mean.cpu().numpy(),
            #                       batch_size)))
            # grad_output_mean = grad_output_mean.float()
            #
            # input_t = input.transpose(0, 1).double()
            # input_size = input_t.size()
            # input_t.resize_(int(input_size[0]), int(np.prod(input_size[1:])))
            # dotP = (input_t - torch.unsqueeze(self.mean.double(), 1)) * \
            #        grad_output_t
            # dotP = dotP.mean(dim=1)
            # if len(device_ids) > 1 and self.sync:
            #     dotP.copy_(torch.from_numpy(
            #         self.cum_mean(grad_output.get_device(),
            #                       dotP.cpu().numpy(),
            #                       batch_size)))
            # dotP = dotP.float()

            batch_size = int(grad_output.size(0))
            grad_output_mean_cuda = grad_output.new().resize_(grad_output.size(1))
            dotP_cuda = grad_output.new().resize_(
                grad_output.size(1))
            batch_norm.BatchNormalizationP_mean_grad_cuda(
                input, grad_output, self.running_mean,
                self.mean, grad_output_mean_cuda, dotP_cuda, self.training
            )
            if len(self.device_ids) > 1 and self.sync:
                grad_output_mean_cuda.copy_(torch.from_numpy(
                    self.cum_mean(grad_output.get_device(),
                                  grad_output_mean_cuda.cpu().numpy(),
                                  batch_size)))
                dotP_cuda.copy_(torch.from_numpy(
                    self.cum_mean(grad_output.get_device(),
                                  dotP_cuda.cpu().numpy(),
                                  batch_size)))

            # pdb.set_trace()

            batch_norm.BatchNormalizationP_backward_cuda(
                input, grad_output, grad_output_mean_cuda, dotP_cuda,
                grad_input, grad_weight, grad_bias,
                weight, self.running_mean, self.running_var,
                self.mean, self.var, self.training, 1, self.eps)
        return grad_input, grad_weight, grad_bias
