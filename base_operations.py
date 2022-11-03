import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TensorConvLayerInput(nn.Module):
    """ tensor based convolution layer """
    def __init__(self, device, kh=3, kw=3, pad_h=0, pad_w=0, stride_h=2, stride_w=2, in_channel=1, out_channel=1, input_tensor=[3, 1], output_tensor=[5, 5, 5, 5], compress_tensor=False):
        super(TensorConvLayerInput, self).__init__()
        assert(pad_h < kh and pad_w < kw)
        self.device = device
        self.kh = kh
        self.kw = kw
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.out_channel = out_channel
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.compress_tensor = compress_tensor   
        self.pad_h = pad_h
        self.pad_w = pad_w

        
        if len(self.output_tensor) == 6 and not compress_tensor:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[1], output_tensor[1], 
						output_tensor[2], output_tensor[3], output_tensor[4], output_tensor[5]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        elif len(self.output_tensor) == 4 and compress_tensor:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[1], input_tensor[0], output_tensor[0], 
						output_tensor[1], output_tensor[2], output_tensor[3]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        elif len(self.output_tensor) == 4 and not compress_tensor:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[1], output_tensor[1], 
						output_tensor[2], output_tensor[3]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        elif len(self.output_tensor) == 7 and not compress_tensor:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[1], output_tensor[1], 
						output_tensor[2], output_tensor[3], output_tensor[4], output_tensor[5], output_tensor[6]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        else:
          raise Exception("output tensor shape is not supported yet")

    def forward(self, x):
        in_tensor_shape = x.size()
        output_h = (in_tensor_shape[2] + 2*self.pad_h - self.kh) // self.stride_h + 1
        output_w = (in_tensor_shape[3] + 2*self.pad_w - self.kw) // self.stride_w + 1
        assert output_h > 0
        assert output_w > 0


        if len(self.output_tensor) == 6 and not self.compress_tensor:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, self.input_tensor[0], self.output_tensor[1], self.output_tensor[2],
				 self.output_tensor[3], self.output_tensor[4], self.output_tensor[5]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j, :, :, :, :, :, :] = torch.einsum('abcdef,gbcdfhijkl->agehijkl', 
                                    x[:, :, h_offset_start:h_offset_end, w_offset_start:w_offset_end, :, :], 
                                    self.weights[:, :, h_weights_offset_start:h_weights_offset_end, w_weights_offset_start:w_weights_offset_end, :, :, :, :, :, :])
        elif len(self.output_tensor) == 4 and self.compress_tensor:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, self.output_tensor[0], self.output_tensor[1], self.output_tensor[2],
				 self.output_tensor[3]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j, :, :, :, :] = torch.einsum('abcdef,gbcdfehijk->aghijk', 
                                    x[:, :, h_offset_start:h_offset_end, w_offset_start:w_offset_end, :, :], 
                                    self.weights[:, :, h_weights_offset_start:h_weights_offset_end, w_weights_offset_start:w_weights_offset_end, :, :, :, :, :, :])
        elif len(self.output_tensor) == 4 and not self.compress_tensor:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, self.input_tensor[0], self.output_tensor[1], self.output_tensor[2],
				 self.output_tensor[3]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j, :, :, :, :] = torch.einsum('abcdef,gbcdfhij->agehij', 
                                    x[:, :, h_offset_start:h_offset_end, w_offset_start:w_offset_end, :, :], 
                                    self.weights[:, :, h_weights_offset_start:h_weights_offset_end, w_weights_offset_start:w_weights_offset_end, :, :, :, :])
        elif len(self.output_tensor) == 7 and not self.compress_tensor:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, self.input_tensor[0], self.output_tensor[1], self.output_tensor[2],
				 self.output_tensor[3], self.output_tensor[4], self.output_tensor[5], self.output_tensor[6]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j, :, :, :, :, :, :, :] = torch.einsum('abcdef,gbcdfhijklm->agehijklm', 
                                    x[:, :, h_offset_start:h_offset_end, w_offset_start:w_offset_end, :, :], 
                                    self.weights[:, :, h_weights_offset_start:h_weights_offset_end, w_weights_offset_start:w_weights_offset_end, :, :, :, :, :, :, :])
        else:
          raise Exception("output tensor shape is not supported yet")

        return out_tensor


class TensorConvLayer4d(nn.Module):
    """ tensor based convolution layer """
    def __init__(self, device, pad_h=0, pad_w=0, kh=3, kw=3, stride_h=2, stride_w=2, in_channel=1, out_channel=1, input_tensor=[3, 3, 3, 3], output_tensor=[3, 3, 3, 3], compress_tensor=False):
        super(TensorConvLayer4d, self).__init__()
        self.device = device
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.kh = kh
        self.kw = kw
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.out_channel = out_channel
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.compress_tensor = compress_tensor   

        if compress_tensor == True:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[3], input_tensor[2], 
                                                 input_tensor[1], input_tensor[0]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        elif len(self.output_tensor) == 4:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[3], input_tensor[2], 
                                                 output_tensor[2], output_tensor[3]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        elif len(self.output_tensor) == 2:
          self.weights = nn.Parameter(torch.empty([out_channel, in_channel, kh, kw, input_tensor[3], input_tensor[2], 
                                                 input_tensor[1], output_tensor[3]], dtype=torch.float32))
          nn.init.kaiming_normal_(self.weights, a=0, mode='fan_out')
        else:
          raise Exception("output tensor shape is not supported yet")
        
    def forward(self, x):
        in_tensor_shape = x.size()
        output_h = (in_tensor_shape[2] + 2*self.pad_h - self.kh) // self.stride_h + 1
        output_w = (in_tensor_shape[3] + 2*self.pad_w - self.kw) // self.stride_w + 1
        assert output_h > 0
        assert output_w > 0
        if self.compress_tensor == True:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w]) 
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    a0, a1, b0, b1, c0, c1, d0, d1 = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j] = torch.einsum('abcdefgh,ibcdhgfe->ai', 
                                    x[:, :, a0:a1, b0:b1, :, :, :, :], 
                                    self.weights[:, :, c0:c1, d0:d1, :, :, :, :])
        elif len(self.output_tensor) == 4:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, 
                                        self.input_tensor[0], self.input_tensor[1], self.output_tensor[2], self.output_tensor[3]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    a0, a1, b0, b1, c0, c1, d0, d1 = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    #print(a0,a1, b0,b1,c0,c1,d0,d1)
                    #print(x[:, :, a0:a1, b0:b1, :, :, :, :].size(), self.weights[:, :, c0:c1, d0:d1, :, :, :, :].size())
                    out_tensor[:, :, i, j, :, :, :, :] = torch.einsum('abcdefgh,ibcdhgjk->aiefjk', 
                                    x[:, :, a0:a1, b0:b1, :, :, :, :], 
                                    self.weights[:, :, c0:c1, d0:d1, :, :, :, :])
        elif len(self.output_tensor) == 2:
            out_tensor = np.zeros([in_tensor_shape[0], self.out_channel, output_h, output_w, self.input_tensor[0], self.output_tensor[3]])
            out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
            for i in range(output_h):
                for j in range(output_w):
                    a0, a1, b0, b1, c0, c1, d0, d1 = cal_offset(i, self.stride_h, self.pad_h, self.kh, j, self.stride_w, self.pad_w, self.kw, in_tensor_shape[2], in_tensor_shape[3])
                    out_tensor[:, :, i, j, :, :] = torch.einsum('abcdefgh,ibcdhgfj->aiej', 
                                    x[:, :, a0:a1, b0:b1, :, :, :, :], 
                                    self.weights[:, :, c0:c1, d0:d1, :, :, :, :])
        else:
          raise Exception("output tensor shape is not supported yet")
        return out_tensor


class TensorBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(TensorBatchNorm, self).__init__()
        self.flatten = nn.Flatten(2)
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        in_tensor_shape = x.size()
        #print(in_tensor_shape)
        x = self.flatten(x)
        x = self.bn(x)
        return x.view(in_tensor_shape)

class TensorBatchNorm1(nn.Module):
    def __init__(self, num_features):
        super(TensorBatchNorm, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        size = x.size()
        x = x.view(size[0], self.num_features, -1)
        x = self.bn(x)
        return x.view(size)

class TensorReLU(nn.Module):
    def __init__(self, device, dim):
        super(TensorReLU, self).__init__()
        self.flatten = nn.Flatten(2)
        self.dim = dim
        self.device = device

    def forward(self, x):
        in_tensor_shape = x.size()
        x = self.flatten(x)
        y = F.max_pool1d(x, kernel_size=self.dim)
        z = torch.where(y>0, 1, 0)
        m = torch.from_numpy(np.ones((in_tensor_shape[0], in_tensor_shape[1], self.dim))).float().to(self.device)
        mask = torch.einsum('abc,abd->abcd', z, m)
        return torch.mul(x.view(in_tensor_shape), mask.view(in_tensor_shape))

class TensorMaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(TensorMaxPool, self).__init__()
        self.flatten = nn.Flatten(2)
        self.kernel_size = kernel_size

    def forward(self, x):
        in_tensor_shape = x.size()
        x = self.flatten(x)
        y = F.max_pool1d(x, kernel_size=self.kernel_size)
        #print(in_tensor_shape)
        #print(y.view(in_tensor_shape[:-4]).size())
        #return y.view(in_tensor_shape[:-4])
        return y.view(in_tensor_shape[:-2])


def cal_offset(i, stride_h, pad_h, kh, j, stride_w, pad_w, kw, h, w):
    h_offset = i*stride_h-pad_h
    w_offset = j*stride_w-pad_w
    h_weights_offset_start = 0
    h_weights_offset_end = kh
    w_weights_offset_start = 0
    w_weights_offset_end = kw
    if h_offset<0:
        h_offset_start = 0
        h_offset_end = h_offset+kh
        h_weights_offset_start = -h_offset
        assert h_weights_offset_start<kh
    elif h_offset+kh > h:
        h_offset_start = h_offset
        h_offset_end = h
        h_weights_offset_end = h_offset_end-h_offset_start
        assert h_weights_offset_end<kh
    else:
        h_offset_start = h_offset
        h_offset_end = h_offset+kh
    
    if w_offset<0:
        w_offset_start = 0
        w_offset_end = w_offset+kh
        w_weights_offset_start = -w_offset
        assert w_weights_offset_start<kw
    elif w_offset+kw > w:
        w_offset_start = w_offset
        w_offset_end = w
        w_weights_offset_end = w_offset_end-w_offset_start
        assert w_weights_offset_end<kw
    else:
        w_offset_start = w_offset
        w_offset_end = w_offset + kw
    #print(i, j, h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end)

    return h_offset_start, h_offset_end, w_offset_start, w_offset_end, h_weights_offset_start, h_weights_offset_end, w_weights_offset_start, w_weights_offset_end 


