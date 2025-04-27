import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class DoubleConv(nn.Module):
    """
    DoubleConv模块
    (convolution => [BN] => ReLU) * 2
    连续两次卷积: 在U-Net网络中,下采样和上采样过程,每一层都会连续进行两次卷积
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            # 另一种写法
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
            # H_out = (H+2p-k)/s +1 = (H+2-3)/1 + 1 = H 分辨率不变
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), # 归一化
            nn.ReLU(inplace=True), # inplace原地激活， max(0,x)

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样模块
    maxpool池化, 进行下采样, 再接DoubleConv模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 2×2池化窗口,尺寸减半, 取窗口内最大值
            DoubleConv(in_channels,out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """
    上采样模块
    上采样方法有两种: 双线性插值bilinear, 反卷积
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            # 放大两倍, mid_channels为输入通道数一半，减少计算量
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # 卷积计算： H = (H+2p-k)/s + 1 = (H + 0 - 2)/2 + 1 = H/2
            # 反卷积计算： H = s(H-1)+2p-k+2 = 2(H-1)+2-2+2 = 2H ，扩大两倍
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1接收上采样数据, x2接收特征融合的数据
        特征融合: 先对小的feature map进行padding, 然后与x2做concat, 通道叠加
        """
        x1 = self.up(x1) # 尺寸扩大两倍

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 确保二者尺寸一致
        # F.pad(x1,[左,右,上,下])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    """
     UNet网络的输出需要根据分割数量, 整合输出通道(若最后的通道为2, 即分类为2的情况)
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv =  nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        # 参数会自动初始化
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)

        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4) # 跳跃连接
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        
        return logits


if __name__ == "__main__":

    net = UNet(n_channels=24,n_classes=8)

    image = torch.randn(2,24,512,512) # 要求的输入是(B,C,H,W)格式

    result = net(image)

    print(result.shape) # 输出是(B,n_classes,H,W)
    
    # 查看要训练的参数
    for name, param in net.named_parameters():
        # value: {param.data}
        print(f"Name: {name} | Shape: {param.shape} | Trainable: {param.requires_grad}")
