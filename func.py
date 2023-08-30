import torch
import torch.nn as nn
import torch.nn.functional as F
from params import mnist_para
import math

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0.0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = (mnist_para.alpha / 2) / (1 + ((mnist_para.alpha * math.pi / 2) * (x)).square()) * grad_output
        return grad_x

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = self._conv_forward(input.flatten(0, 1).contiguous(), self.weight)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__(
            num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        output = []
        for t in range(input.shape[1]):
            output.append(F.batch_norm(input[:,t,...],
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps))
        return torch.stack(output, dim=1)

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_max_pool2d(input.flatten(0, 1).contiguous(), self.output_size, self.return_indices)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_avg_pool2d(input.flatten(0, 1).contiguous(), self.output_size)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class LIF(nn.Module):
    def __init__(self, train, thresh, tau, heterogeneity):
        super(LIF, self).__init__()
        self.act = atan.apply
        self.tau = tau
        self.thresh = nn.Parameter(heterogeneity*torch.randn_like(thresh)/100+thresh, requires_grad=train)
    def forward(self, input):
        T = input.shape[1]
        mem = 0
        spike_pot = []
        for t in range(T):
            mem = mem * self.tau + input[:, t, ...]
            spike = self.act(mem - self.thresh)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

class Dropout(nn.Dropout):
    def __init__(self, p = 0.5, inplace = False):
        super(Dropout, self).__init__(p, inplace)
    def forward(self, input):
        neuron = list([])
        for t in range(input.shape[1]):
            neuron.append(F.dropout(input[:, t, ...], self.p, self.training, self.inplace))
        return torch.stack(neuron, dim=1)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.linear(input.flatten(0, 1).contiguous(), self.weight, self.bias)
        N = output.shape[-1]
        output = output.view([B,T,N])
        return output


