import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import torch.nn.init as init


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SepnetLinear(nn.Module):
    def __init__(self, A_W, A_H, B_W, B_H, activation=None, batch_normalization=False, kernel_initializer='he_normal',
                 **kwargs):
        super(SepnetLinear, self).__init__(**kwargs)
        # self.A_H = Parameter(A_H)
        # self.A_W = Parameter(A_W)
        # self.B_H = Parameter(B_H)
        # self.B_W = Parameter(B_W)
        # self.A_shape = Parameter(A_size)
        # self.B_shape = Parameter(B_size)
        # self.bias_shape = [A_size[0], B_size[1]]
        self.activation = activation
        self.bias = None
        self.batch_normalization = batch_normalization
        # self.A_ref = Parameter(torch.Tensor(self.A_shape[0], self.A_shape[1]))
        # self.B_ref = Parameter(torch.Tensor(self.B_shape[0], self.B_shape[1]))
        self.A_ref = Parameter(torch.Tensor(A_W, A_H))
        self.B_ref = Parameter(torch.Tensor(B_W, B_H))
        # self.popup_scores_A = Parameter(torch.Tensor(self.A_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_A, a=math.sqrt(5))
        # self.popup_scores_B = Parameter(torch.Tensor(self.B_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_B, a=math.sqrt(5))

        # self.A_ref = Parameter(torch.Tensor(A_size))
        # self.B_ref = Parameter(torch.Tensor(B_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.A_ref, a=math.sqrt(5))
        init.kaiming_uniform_(self.B_ref, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # print(x.shape)
        assert len(x.shape) == 3, 'The input Tensor should have shape=[None, input_height*input_width, input_channels]'
        # adj_A = GetSubnet.apply(self.popup_scores_A.abs(), self.k)
        # adj_B = GetSubnet.apply(self.popup_scores_B.abs(), self.k)
        # self.ref_A = self.A_ref * adj_A
        # self.ref_B = self.B_ref * adj_B
        self.batch_size = x.shape[0]
        self.input_channels = x.shape[1]
        self.input_hw = x.shape[2]

        # print(self.input_hw,self.input_channels)
        # sub matrix A and B
        # self.A = torch.FloatTensor(self.batch_size,self.A_shape[0],self.A_shape[1]).to(device)
        # self.B = torch.FloatTensor(self.batch_size,self.B_shape[0],self.B_shape[1]).to(device)
        # self.bias = self.none
        self.A = self.A_ref.repeat(self.batch_size, 1, 1)
        self.B = self.B_ref.repeat(self.batch_size, 1, 1)

        temp = torch.bmm(x, self.A.permute(0, 2, 1))
        temp = temp.permute(0, 2, 1)

        out = torch.bmm(temp, self.B)
        # out = out.permute(0,2,1)

        # print(self.A.shape, self.B.shape)
        # return out, self.A, self.B

        return out, self.A_ref, self.B_ref


class SepnetLinear_QKV(nn.Module):
    def __init__(self, A_W, A_H, B_W, B_H, activation=None, batch_normalization=False, kernel_initializer='he_normal',
                 **kwargs):
        super(SepnetLinear_QKV, self).__init__(**kwargs)
        # self.A_H = Parameter(A_H)
        # self.A_W = Parameter(A_W)
        # self.B_H = Parameter(B_H)
        # self.B_W = Parameter(B_W)
        # self.A_shape = Parameter(A_size)
        # self.B_shape = Parameter(B_size)
        # self.bias_shape = [A_size[0], B_size[1]]
        self.activation = activation
        self.bias = None
        self.batch_normalization = batch_normalization
        # self.A_ref = Parameter(torch.Tensor(self.A_shape[0], self.A_shape[1]))
        # self.B_ref = Parameter(torch.Tensor(self.B_shape[0], self.B_shape[1]))
        self.A_ref = Parameter(torch.Tensor(A_W, A_H))  # 对矩阵A_ref的一个初始化,大小为(A_W, A_H)
        self.B_ref = Parameter(torch.Tensor(B_W, B_H))
        # self.popup_scores_A = Parameter(torch.Tensor(self.A_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_A, a=math.sqrt(5))
        # self.popup_scores_B = Parameter(torch.Tensor(self.B_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_B, a=math.sqrt(5))

        # self.A_ref = Parameter(torch.Tensor(A_size))
        # self.B_ref = Parameter(torch.Tensor(B_size))
        self.reset_parameters()

    def reset_parameters(self):  # 权重参数和偏置参数的初始化方法
        init.kaiming_uniform_(self.A_ref, a=math.sqrt(5))
        init.kaiming_uniform_(self.B_ref, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_prune_rate(self, k):
        self.k = k
        # if k >=0.01:
        # self.k = 1
        # else:
        # self.k = k*100
        # self.kernel_initializer = ini

    def forward(self, x):
        # print(x.shape)
        assert len(
            x.shape) == 4, 'The input Tensor should have shape=[Batch_size, patch_number+1 ,input_height*input_width, input_channels]'
        # adj_A = GetSubnet.apply(self.popup_scores_A.abs(), self.k)
        # = GetSubnet.apply(self.popup_scores_B.abs(), self.k)
        # self.ref_A = self.A_ref * adj_A
        # self.ref_B = self.B_ref * adj_B
        self.batch_size = x.shape[0]
        self.N = x.shape[1]

        # print(self.input_hw,self.input_channels)
        # sub matrix A and B
        # self.A = torch.FloatTensor(self.batch_size,self.A_shape[0],self.A_shape[1]).to(device)
        # self.B = torch.FloatTensor(self.batch_size,self.B_shape[0],self.B_shape[1]).to(device)
        # self.bias = self.none
        self.A = self.A_ref.repeat(self.batch_size, self.N, 1, 1)  # 将矩阵的维度扩展到4维张量  [b, n, w, h] w=32 ,h=24
        self.B = self.B_ref.repeat(self.batch_size, self.N, 1, 1)  # x = (b,n,24,32)

        # print('aaa',self.A.permute(0, 2, 1).shape)
        temp = torch.matmul(x, self.A.permute(0, 1, 3, 2))
        temp = temp.permute(0, 1, 3, 2)
        # temp = torch.bmm(x, self.A.permute(0,2,1))
        # temp = temp.view(0, 2, 1)

        out = torch.matmul(temp, self.B)
        # out = out.permute(0,2,1)

        # print(self.A.shape, self.B.shape)
        # return out, self.A, self.B

        return out


class SepnetLinear_VIT(nn.Module):
    def __init__(self, A_W, A_H, B_W, B_H, activation=None, batch_normalization=False, kernel_initializer='he_normal',
                 **kwargs):
        super(SepnetLinear_VIT, self).__init__(**kwargs)
        # self.A_H = Parameter(A_H)
        # self.A_W = Parameter(A_W)
        # self.B_H = Parameter(B_H)
        # self.B_W = Parameter(B_W)
        # self.A_shape = Parameter(A_size)
        # self.B_shape = Parameter(B_size)
        # self.bias_shape = [A_size[0], B_size[1]]
        self.activation = activation
        self.bias = None
        self.batch_normalization = batch_normalization
        # self.A_ref = Parameter(torch.Tensor(self.A_shape[0], self.A_shape[1]))
        # self.B_ref = Parameter(torch.Tensor(self.B_shape[0], self.B_shape[1]))
        self.A_ref = Parameter(torch.Tensor(A_W, A_H))  # 对矩阵A_ref的一个初始化,大小为(A_W, A_H)
        self.B_ref = Parameter(torch.Tensor(B_W, B_H))
        # self.popup_scores_A = Parameter(torch.Tensor(self.A_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_A, a=math.sqrt(5))
        # self.popup_scores_B = Parameter(torch.Tensor(self.B_ref.shape))
        # nn.init.kaiming_uniform_(self.popup_scores_B, a=math.sqrt(5))

        # self.A_ref = Parameter(torch.Tensor(A_size))
        # self.B_ref = Parameter(torch.Tensor(B_size))
        self.reset_parameters()

    def reset_parameters(self):  # 权重参数和偏置参数的初始化方法
        init.kaiming_uniform_(self.A_ref, a=math.sqrt(5))
        init.kaiming_uniform_(self.B_ref, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_prune_rate(self, k):
        self.k = k
        # if k >=0.01:
        # self.k = 1
        # else:
        # self.k = k*100
        # self.kernel_initializer = ini

    def forward(self, x):
        # print(x.shape)
        assert len(
            x.shape) == 4, 'The input Tensor should have shape=[Batch_size, patch_number+1 ,input_height*input_width, input_channels]'
        # adj_A = GetSubnet.apply(self.popup_scores_A.abs(), self.k)
        # = GetSubnet.apply(self.popup_scores_B.abs(), self.k)
        # self.ref_A = self.A_ref * adj_A
        # self.ref_B = self.B_ref * adj_B
        self.batch_size = x.shape[0]
        self.N = x.shape[1]

        # print(self.input_hw,self.input_channels)
        # sub matrix A and B
        # self.A = torch.FloatTensor(self.batch_size,self.A_shape[0],self.A_shape[1]).to(device)
        # self.B = torch.FloatTensor(self.batch_size,self.B_shape[0],self.B_shape[1]).to(device)
        # self.bias = self.none
        self.A = self.A_ref.repeat(self.batch_size, self.N, 1, 1)  # 将矩阵的维度扩展到4维张量  [b, n, w, h] w=32 ,h=24
        self.B = self.B_ref.repeat(self.batch_size, self.N, 1, 1)  # x = (b,n,24,32)

        # print('aaa',self.A.permute(0, 2, 1).shape)
        temp = torch.matmul(x, self.A.permute(0, 1, 3, 2))  # x*(b,n,h,w)
        temp = temp.permute(0, 1, 3, 2)
        # temp = torch.bmm(x, self.A.permute(0,2,1))
        # temp = temp.view(0, 2, 1)

        out = torch.matmul(temp, self.B)
        out = out.permute(0, 1, 3, 2)

        # print(self.A.shape, self.B.shape)
        # return out, self.A, self.B

        return out


class SepnetLinear_200(nn.Module):
    def __init__(self, A_W, A_H, B_W, B_H, activation=None, batch_normalization=False, kernel_initializer='he_normal',
                 **kwargs):
        super(SepnetLinear_200, self).__init__(**kwargs)
        # self.A_H = Parameter(A_H)
        # self.A_W = Parameter(A_W)
        # self.B_H = Parameter(B_H)
        # self.B_W = Parameter(B_W)
        # self.A_shape = Parameter(A_size)
        # self.B_shape = Parameter(B_size)
        # self.bias_shape = [A_size[0], B_size[1]]
        self.activation = activation
        self.bias = None
        self.batch_normalization = batch_normalization
        # self.A_ref = Parameter(torch.Tensor(self.A_shape[0], self.A_shape[1]))
        # self.B_ref = Parameter(torch.Tensor(self.B_shape[0], self.B_shape[1]))
        self.A_ref = Parameter(torch.Tensor(A_W, A_H))  # 对矩阵A_ref的一个初始化,大小为(A_W, A_H)
        self.B_ref = Parameter(torch.Tensor(B_W, B_H))
        self.popup_scores_A = Parameter(torch.Tensor(self.A_ref.shape))
        nn.init.kaiming_uniform_(self.popup_scores_A, a=math.sqrt(5))
        self.popup_scores_B = Parameter(torch.Tensor(self.B_ref.shape))
        nn.init.kaiming_uniform_(self.popup_scores_B, a=math.sqrt(5))

        # self.A_ref = Parameter(torch.Tensor(A_size))
        # self.B_ref = Parameter(torch.Tensor(B_size))
        self.reset_parameters()

    def reset_parameters(self):  # 权重参数和偏置参数的初始化方法
        init.kaiming_uniform_(self.A_ref, a=math.sqrt(5))
        init.kaiming_uniform_(self.B_ref, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_prune_rate(self, k):
        self.k = k * 100
        # self.kernel_initializer = ini

    def forward(self, x):
        # print(x.shape)
        assert len(x.shape) == 3, 'The input Tensor should have shape=[None, input_height*input_width, input_channels]'
        adj_A = GetSubnet.apply(self.popup_scores_A.abs(), self.k)
        adj_B = GetSubnet.apply(self.popup_scores_B.abs(), self.k)
        self.ref_A = self.A_ref * adj_A
        self.ref_B = self.B_ref * adj_B
        self.batch_size = x.shape[0]
        self.input_channels = x.shape[1]
        self.input_hw = x.shape[2]

        # print(self.input_hw,self.input_channels)
        # sub matrix A and B
        # self.A = torch.FloatTensor(self.batch_size,self.A_shape[0],self.A_shape[1]).to(device)
        # self.B = torch.FloatTensor(self.batch_size,self.B_shape[0],self.B_shape[1]).to(device)
        # self.bias = self.none
        self.A = self.ref_A.repeat(self.batch_size, 1, 1)  # 将矩阵的维度扩展到三维张量
        self.B = self.ref_B.repeat(self.batch_size, 1, 1)

        # print('aaa',self.A.permute(0, 2, 1).shape)
        temp = torch.bmm(x, self.A.permute(0, 2, 1))
        temp = temp.permute(0, 2, 1)
        # temp = torch.bmm(x, self.A.permute(0,2,1))
        # temp = temp.view(0, 2, 1)

        out = torch.bmm(temp, self.B)
        # out = out.permute(0,2,1)

        # print(self.A.shape, self.B.shape)
        # return out, self.A, self.B

        return out, self.A_ref, self.B_ref


class Linear_ARLST(nn.Module):
    def __init__(self, input_dim, output_dim, input_dim_A, input_dim_B, output_dim_A, output_dim_B, activation=None,
                 bias=True, batch_normalization=False, **kwargs):
        super(Linear_ARLST, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_A = input_dim_A
        self.input_dim_B = input_dim_B
        self.output_dim_A = output_dim_A
        self.output_dim_B = output_dim_B

        self.A = Parameter(torch.empty(input_dim_A, output_dim_A))
        self.B = Parameter(torch.empty(input_dim_B, output_dim_B))
        self.popup_scores_A = Parameter(torch.empty(self.A.shape))
        self.popup_scores_B = Parameter(torch.empty(self.B.shape))
        nn.init.kaiming_uniform_(self.popup_scores_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.popup_scores_B, a=math.sqrt(5))

        # self.activation = activation
        self.A.requires_grad = False
        self.B.requires_grad = False
        self.A_ref = 0
        self.B_ref = 0

        #
        self.use_bias = bias
        self.bias = Parameter(torch.Tensor(output_dim))
        self.bias.requires_grad = False
        # self.batch_normalization = batch_normalization

        # self.reset_parameters()

    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.A, a=math.sqrt(5))
    #     init.kaiming_uniform_(self.B, a=math.sqrt(5))
    #     if self.use_bias:
    #         bound = 1 / math.sqrt(self.input_dim)
    #         init.uniform_(self.bias, -bound, bound)

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        assert self.input_dim == (
                self.input_dim_A * self.input_dim_B), "input_dim must be equal to input_dim_L * input_dimH"
        assert self.output_dim == (
                self.output_dim_A * self.output_dim_B), "output_dim must be equal to output_dim_L * output_dimH"

        adj_A = GetSubnet.apply(self.popup_scores_A.abs(), self.k)
        adj_B = GetSubnet.apply(self.popup_scores_B.abs(), self.k)
        self.A_ref = self.A * adj_A
        self.B_ref = self.B * adj_B

        new_shape = x.size()[:-1] + (self.input_dim_A, self.input_dim_B,)
        x = x.view(*new_shape)
        x = x.matmul(self.B_ref)
        new_shape = x.size()[:-2] + (self.output_dim_B, self.input_dim_A,)
        x = x.view(*new_shape)
        x = x.matmul(self.A_ref)
        new_shape = x.size()[:-2] + (self.output_dim_B * self.output_dim_A,)
        if self.use_bias:
            out = x.view(new_shape) + self.bias
        else:
            out = x.view(new_shape)

        return out, self.A_ref, self.B_ref
