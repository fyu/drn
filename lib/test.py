import pdb
import time
import logging

import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

from modules import batchnormsync

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batchnormsync.BatchNormSync.checking_mode = True
batchnormsync.BatchNormSync.sync = True

cuda = True
batch_size = 3
input = torch.randn(3, 3, 2, 2).float()
# input = torch.Tensor(range(60 * batch_size)).float().resize_(batch_size, 3, 2, 2) / 100
bn = batchnormsync.BatchNormSync(3, eps=0, affine=True,
                                 device_ids=None)
bn2 = torch.nn.BatchNorm2d(3, eps=0, affine=False)
# bn.train()

bn1 = batchnormsync.BatchNormSync(3, eps=0, affine=True, device_ids=[0])

bn1.train()

if cuda:
    bn = torch.nn.DataParallel(bn)
    bn2 = torch.nn.DataParallel(bn2)

    bn = bn.cuda()
    bn1 = bn1.cuda()
    bn2 = bn2.cuda()
    input = input.cuda()


inputs = (Variable(input, requires_grad=True),)
# output = bn(inputs[0])

# output1 = bn1(inputs[0])
# output2 = bn2(inputs[0])
# print((output1 - output2).abs().max())
# print((output - output2).abs().max())
# test = gradcheck(bn, inputs, eps=1e-4, atol=1e-4, rtol=1e-8)
for i in range(1000):
    logger.info(i)
    start_time = time.time()
    test = gradcheck(bn, inputs, eps=1e-4, atol=1e-2, rtol=1e-3)
    logger.info('%s %f', test, time.time() - start_time)
