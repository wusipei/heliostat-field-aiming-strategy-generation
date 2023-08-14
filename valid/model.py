import torch
from torch import nn

#from AE_train_new_conv_try.py
class AE_conv(nn.Module):
    def __init__(self, inputdim=8):
        super(AE_conv, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(inputdim, 8, (5,4), stride=(2,1), padding=(0,2)),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(8)
            )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, (3,3), stride=(2,1), padding=(0,1)),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(16)
            )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 16, (3,3), stride=(2,2), padding=(1,1)),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(16)
            )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(16, 8, (3,3), stride=(2,2), padding=(1,1)),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(8)
            )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(8, 3, (3,3), stride=(2,2), padding=(1,1)),
            nn.Sigmoid(),
            )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(3, 8, 4, 1, 1),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(8)
            )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, 2, 1),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(16)
            )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, 2, 0),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(16)
            )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (6,3), (2,1), (1,1)),
            nn.Sigmoid(),
            torch.nn.BatchNorm2d(8)
            )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(8, inputdim, (6,2), (2,1), (0,1)),
            nn.Sigmoid(),
            )
    def normal_weight_init(self, mean=0.5, std=0.5):
        for m in self.children():
            torch.nn.init.normal_(m[0].weight)
        
    def forward(self, x):
        x = self.encoder1(x)
        #print(x.shape)
        x = self.encoder2(x)
        #print(x.shape)
        x = self.encoder3(x)
        #print(x.shape)
        x = self.encoder4(x)
        #print(x.shape)
        x = self.encoder5(x)
        #print(x.shape)
        encode = x[:]
        
        x = self.decoder1(x)
        #print(x.shape)
        x = self.decoder2(x)
        #print(x.shape)
        x = self.decoder3(x)
        #print(x.shape)
        x = self.decoder4(x)
        #print(x.shape)
        x = self.decoder5(x)
        #print(x.shape)
        decode = x[:]
        return encode, decode


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=(3,3), stride=1, padding=(1,1), groups=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, groups=groups)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=(3,3), stride=1, padding=(1,1), groups=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding,groups=groups)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        #self.relu = torch.nn.ReLU(True)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter
                               ,activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv9 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv10 = ConvBlock(num_filter * 8, num_filter * 8)
        
        self.conv11 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=False)
        
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=False)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=False)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv6 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv7 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        
        self.deconv8 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv9 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv10 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv11 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)
               
        
    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        
        enc8 = self.conv8(enc7)
        enc9 = self.conv9(enc8)
        enc10 = self.conv10(enc9)
        
        enc11 = self.conv11(enc10)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc11)
        dec1 = torch.cat([dec1, enc10], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc9], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc8], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc7], 1)
        
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc6], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc5], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc4], 1)
        
        
        dec8 = self.deconv8(dec7)
        dec8 = torch.cat([dec8, enc3], 1)
        dec9 = self.deconv9(dec8)
        dec9 = torch.cat([dec9, enc2], 1)
        dec10 = self.deconv10(dec9)
        dec10 = torch.cat([dec10, enc1], 1)
        
        dec11 = self.deconv11(dec10)
        #out = torch.nn.Tanh()(dec8)
        out = torch.nn.Sigmoid()(dec11)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal_(m.deconv.weight, mean, std)

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv9 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv10 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)




