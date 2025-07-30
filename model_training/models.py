import torch
import torch.nn as nn
import timm

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_tensor):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_tensor], dim=1)
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, msi_channels, swin_model_name):
        super().__init__()
        self.encoder = timm.create_model(
            swin_model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3)
        )
        encoder_channels = self.encoder.feature_info.channels()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[3], encoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[3]),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = UNetDecoderBlock(encoder_channels[3], encoder_channels[2], 256)
        self.decoder2 = UNetDecoderBlock(256, encoder_channels[1], 128)
        self.decoder3 = UNetDecoderBlock(128, encoder_channels[0], 64)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, msi_channels, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self, rgb_img):
        encoder_features_cl = self.encoder(rgb_img)
        encoder_features_cf = [feat.permute(0, 3, 1, 2) for feat in encoder_features_cl]
        f0, f1, f2, f3 = encoder_features_cf
        b = self.bottleneck(f3)
        d1 = self.decoder1(b, f2)
        d2 = self.decoder2(d1, f1)
        d3 = self.decoder3(d2, f0)
        d4 = self.final_up(d3)
        d4_upsampled = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.final_conv(d4_upsampled)
        return self.tanh(output)

class Discriminator(nn.Module):
    def __init__(self, msi_channels, img_size):
        super().__init__()
        def block(i, o, bn=True):
            layers = [nn.Conv2d(i, o, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn: layers.append(nn.BatchNorm2d(o, 0.8))
            return layers
        ds_size = img_size // 16
        self.model = nn.Sequential(*block(msi_channels, 64, bn=False), *block(64, 128), *block(128, 256), *block(256, 512))
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size**2, 1))
        
    def forward(self, img):
        out = self.model(img)
        out = out.reshape(img.shape[0], -1)
        return self.adv_layer(out)