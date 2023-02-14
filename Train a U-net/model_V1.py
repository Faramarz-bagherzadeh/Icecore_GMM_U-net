
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(0)


class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()

        # Down Sampling
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')

        self.conv3 = nn.Conv2d(32, 52, 3, padding='same')
        self.conv4 = nn.Conv2d(52, 52, 3, padding='same')

        self.conv5 = nn.Conv2d(52, 72, 3, padding='same')
        self.conv6 = nn.Conv2d(72, 72, 3, padding='same')

        self.conv7 = nn.Conv2d(72, 92, 3, padding='same')
        self.conv8 = nn.Conv2d(92, 92, 3, padding='same')

        self.conv9 = nn.Conv2d(92, 112, 3, padding='same')
        self.conv10 = nn.Conv2d(112, 112, 3, padding='same')

        self.conv11 = nn.Conv2d(112, 128, 3, padding='same')
        self.conv12 = nn.Conv2d(128, 128, 3, padding='same')


        # Up Sampling
        # for up sampling block we need a upsample and Concat and convolution
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_conv = nn.Conv2d(240, 120, 3, padding='same')

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_conv  = nn.Conv2d(212, 106, 3, padding='same')

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_conv = nn.Conv2d(178, 90, 3, padding='same')

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_conv = nn.Conv2d(142, 70, 3, padding='same')

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up5_conv = nn.Conv2d(102, 50, 3, padding='same')

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up6_conv = nn.Conv2d(82, 32, 3, padding='same')

        self.output = nn.Conv2d(32, 1, 3, padding='same')


    def forward(self, x):
        # Down sampling
        x0 = self.conv1(x)
        x1 = F.max_pool2d(F.relu(x0), (2, 2))
        x2 = self.conv2(x1)

        x3 = F.max_pool2d(F.relu(self.conv3(x2)), (2, 2))
        x4 = self.conv4(x3)

        x5 = F.max_pool2d(F.relu(self.conv5(x4)), (2, 2))
        x6 = self.conv6(x5)

        x7 = F.max_pool2d(F.relu(self.conv7(x6)), (2, 2))
        x8 = self.conv8(x7)

        x9 = F.max_pool2d(F.relu(self.conv9(x8)), (2, 2))
        x10 = self.conv10(x9)

        x11 = F.max_pool2d(F.relu(self.conv11(x10)), (2, 2))
        x12 = self.conv12(x11)


        # Up sampling
        x = self.up1(x12)
        x = torch.cat([x, x10], dim=1)
        x = F.relu(self.up1_conv(x))

        x = self.up2(x)
        x = torch.cat([x, x8], dim=1)
        x = F.relu(self.up2_conv(x))

        x = self.up3(x)
        x = torch.cat([x, x6], dim=1)
        x = F.relu(self.up3_conv(x))

        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.up4_conv(x))

        x = self.up5(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.up5_conv(x))

        x = self.up6(x)
        x = torch.cat([x, x0], dim=1)
        x = F.relu(self.up6_conv(x))
        x = self.output(x)
        #x = torch.sigmoid(self.output(x))

        return x


def model():
    mm = UNET()
    print(model)
    total_params = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    print ('********')
    print('Total trainable parameters = ',total_params)
    return mm
# Press the green button in the gutter to run the script.


'''if __name__ == '__main__':
    model = UNET()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ('********')
    print('Total trainable parameters = ',total_params)

    a = torch.rand(2,1,2048,2048)
    b = model(a)
    print ('model output ' , b.shape)

'''