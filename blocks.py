import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision



class simple_block(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=1, 
                groups=4, 
                is_down=True):
        super(simple_block, self).__init__()

        if is_down:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, 
                            out_channels//2, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=groups),
                nn.BatchNorm3d(num_features=out_channels//2),
                #nn.InstanceNorm3d(out_channels//2, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv3d(out_channels//2, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=groups),
                nn.BatchNorm3d(num_features=out_channels),
                #nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=groups),
                nn.BatchNorm3d(num_features=out_channels),
                #nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv3d(out_channels, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=groups),
                nn.BatchNorm3d(num_features=out_channels),
                #nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

class Up_sample(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=2, 
                padding=1, 
                output_padding=0, 
                groups=2):
        super(Up_sample, self).__init__()

        self.block = nn.ConvTranspose3d(in_channels, 
                                        out_channels, 
                                        kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        output_padding=output_padding,
                                        groups=groups)
        
    def forward(self, x):
        return self.block(x)

class Down_sample(nn.Module):
    def __init__(self, kernel_size, stride=2, padding=1):
        super(Down_sample, self).__init__()

        self.block = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        return self.block(x)

            



