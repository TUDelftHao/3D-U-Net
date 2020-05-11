import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample
from torchviz import make_dot
import torch.autograd as autograd
from torch.autograd import Variable
from utils import count_params, load_config
from torchsummary import summary


# simplest U-Net
class init_U_Net(nn.Module):

    def __init__(self, 
                input_modalites, 
                output_channels, 
                base_channel, 
                softmax=True):

        self.softmax = softmax
        super(init_U_Net, self).__init__()

        self.min_channel = base_channel
        self.down_conv1  = simple_block(input_modalites, self.min_channel*2, 3)
        self.down_sample_1 = Down_sample(2)
        self.down_conv2  = simple_block(self.min_channel*2, self.min_channel*4, 3)
        self.down_sample_2 = Down_sample(2)
        self.down_conv3  = simple_block(self.min_channel*4, self.min_channel*8, 3,)
        self.down_sample_3 = Down_sample(2)

        self.bridge = simple_block(self.min_channel*8, self.min_channel*16, 3)
        
        self.up_sample_1   = Up_sample(self.min_channel*16, self.min_channel*16, 3) # change here to 3 if crop size is 128
        self.up_conv1  = simple_block(self.min_channel*24, self.min_channel*8, 3, is_down=False)
        self.up_sample_2   = Up_sample(self.min_channel*8, self.min_channel*8, 3)
        # change here to 3 if crop size is 128
        self.up_conv2  = simple_block(self.min_channel*12, self.min_channel*4, 3, is_down=False)
        self.up_sample_3   = Up_sample(self.min_channel*4, self.min_channel*4, 2) 
        self.up_conv3  = simple_block(self.min_channel*6, self.min_channel*2, 3, is_down=False)

        self.out = nn.Conv3d(self.min_channel*2, output_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # encoder path

        self.block_1 = self.down_conv1(x)
        self.block_1_pool = self.down_sample_1(self.block_1)
        self.block_2 = self.down_conv2(self.block_1_pool)
        self.block_2_pool = self.down_sample_2(self.block_2)
        self.block_3 = self.down_conv3(self.block_2_pool)
        self.block_3_pool = self.down_sample_3(self.block_3)

        # bridge
        self.block_4 = self.bridge(self.block_3_pool)

        # decoder path

        self.block_5_upsample = self.up_sample_1(self.block_4)
        self.concat = torch.cat([self.block_5_upsample, self.block_3], dim=1)
        self.block_5 = self.up_conv1(self.concat)
        self.block_6_upsample = self.up_sample_2(self.block_5)
        self.concat = torch.cat([self.block_6_upsample, self.block_2], dim=1)
        self.block_6 = self.up_conv2(self.concat)
        self.block_7_upsample = self.up_sample_3(self.block_6)
        self.concat = torch.cat([self.block_7_upsample, self.block_1], dim=1)
        self.block_7 = self.up_conv3(self.concat)

        res = self.out(self.block_7)

        if self.softmax:
            res = F.softmax(res, dim=1)

        return res


if __name__ == '__main__':
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = int(config['PARAMETERS']['base_channels'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = init_U_Net(input_modalites, output_channels, base_channel)
    net.to(device)

    # print(net)
    print(net.block_4.shape)

    # params = list(net.parameters())
    # for i in range(len(params)):
    #     layer_shape = params[i].size()
        
    #     print(len(layer_shape))

    # print parameters infomation
    # count_params(net)
    # input = torch.randn(2, 4, 98, 98, 98).to(device)
    # input = torch.randn(1, 4, 128, 128, 128).to(device)
    # y = net(input)
    # print(y.shape)
    # print(np.unique(y.detach().cpu().numpy()))
    # summary(net, input_size=(4, 98, 98, 98))

    # count_intermidiate_size(model=net, input=input)


        


