import torch.nn as nn
import torch
import torch.nn.functional as F
from base_operations import TensorConvLayerInput, TensorConvLayer4d TensorBatchNorm 


class TCNN3_2(nn.Module):
    def __init__(self, device):
        super(TCNN3_2, self).__init__()

        self.features1 = nn.Sequential(
            TensorConvLayerInput(device, kh=3, kw=3, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[3, 1], output_tensor=[6, 6, 6, 6], compress_tensor=True),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )

        self.block1 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )

        self.features2 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, stride_h=2, stride_w=2, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )
        self.block2 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )
        self.features3 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )
        self.block3 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
            nn.PReLU(),
            TensorConvLayer4d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6]),
            TensorBatchNorm(1),
        )

        self.features4 = nn.Sequential(
            TensorConvLayer4d(device, kh=3, kw=3, in_channel=1, out_channel=10, input_tensor=[6, 6, 6, 6], output_tensor=[6, 6, 6, 6], compress_tensor=True),
            TensorBatchNorm(10),
            nn.PReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1, end_dim=- 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1) 
        x = x.unsqueeze(5)
        x1 = self.features1(x)
        b1 = self.block1(x1)
        x2 = self.features2(x1+b1)
        b2 = self.block2(x2)
        x3 = self.features3(x2+b2)
        b3 = self.block3(x3)
        x4 = self.features4(x3+b3)
        x = self.flatten(x4)
        output = F.log_softmax(x, dim=1)
        return output


