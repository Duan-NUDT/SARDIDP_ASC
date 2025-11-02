import torch.nn as nn

from unet_parts import inconv, down, up


class Unet_single(nn.Module):
    def __init__(self, feature_num=64):
        super(Unet_single, self).__init__()

        self.inc = inconv(1, feature_num)

        self.down1 = down(feature_num, feature_num * 2)
        self.down2 = down(feature_num * 2, feature_num * 4)
        self.down3 = down(feature_num * 4, feature_num * 4)


        self.up1 = up(feature_num * 8, feature_num * 2)
        self.up2 = up(feature_num * 4, feature_num * 1)
        self.up3 = up(feature_num * 2, feature_num)

        # self.down1 = down(feature_num, feature_num*2)
        # self.down2 = down(feature_num*2, feature_num*4)
        # self.down3 = down(feature_num*4, feature_num*4)
        #
        #
        # self.up1 = up(feature_num*8, feature_num*2)
        # self.up2 = up(feature_num*4, feature_num*1)
        # self.up3 = up(feature_num*2, feature_num)

        self.outc = nn.Sequential(
        nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(feature_num, 1, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, input):
        fi = self.inc(input)
        x2 = self.down1(fi)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        ff3 = self.up3(x, fi)

        output = self.outc(ff3)
        return output