import torch
from timm.models.resnet import resnet50d, resnet101d
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.deeplabv3 import ASPP


class RegularStream(nn.Module):
    def __init__(self, backbone_type='resnet50'):
        super().__init__()
        if backbone_type == 'resnet50':
            self.backbone = resnet50d(output_stride=8)
        else:
            self.backbone = resnet101d(output_stride=8)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        res0 = self.backbone.maxpool(x)

        res1 = self.backbone.layer1(res0)
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)

        return x, res1, res2, res3, res4


class ShapeStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.res2_conv = nn.Conv2d(512, 1, 1)
        self.res3_conv = nn.Conv2d(1024, 1, 1)
        self.res4_conv = nn.Conv2d(2048, 1, 1)
        self.res1 = BasicBlock(64, 64, 1)
        self.res2 = BasicBlock(32, 32, 1)
        self.res3 = BasicBlock(16, 16, 1)
        self.res1_pre = nn.Conv2d(64, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32)
        self.gate2 = GatedConv(16, 16)
        self.gate3 = GatedConv(8, 8)
        self.gate = nn.Conv2d(8, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, x, res2, res3, res4, grad):
        size = grad.size()[-2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        res2 = F.interpolate(self.res2_conv(res2), size, mode='bilinear', align_corners=True)
        res3 = F.interpolate(self.res3_conv(res3), size, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(res4), size, mode='bilinear', align_corners=True)

        gate1 = self.gate1(self.res1_pre(self.res1(x)), res2)
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res3)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res4)
        gate = torch.sigmoid(self.gate(gate3))
        feat = torch.sigmoid(self.fuse(torch.cat((gate, grad), dim=1)))
        return gate, feat


class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out


class FeatureFusion(ASPP):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256):
        # atrous_rates (6, 12, 18) is for stride 16
        super().__init__(in_channels, atrous_rates, out_channels)
        self.shape_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.project = nn.Conv2d((len(atrous_rates) + 3) * out_channels, out_channels, 1, bias=False)
        self.fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

    def forward(self, res1, res4, feat):
        res = []
        for conv in self.convs:
            res.append(conv(res4))
        res = torch.cat(res, dim=1)
        feat = F.interpolate(feat, res.size()[-2:], mode='bilinear', align_corners=True)
        res = torch.cat((res, self.shape_conv(feat)), dim=1)
        coarse = F.interpolate(self.project(res), res1.size()[-2:], mode='bilinear', align_corners=True)
        fine = self.fine(res1)
        out = torch.cat((coarse, fine), dim=1)
        return out


class GatedSCNN(nn.Module):
    def __init__(self, backbone_type='resnet50', num_classes=19):
        super().__init__()

        self.regular_stream = RegularStream(backbone_type)
        self.shape_stream = ShapeStream()
        self.feature_fusion = FeatureFusion(2048, (12, 24, 36), 256)
        self.seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, grad):
        x, res1, res2, res3, res4 = self.regular_stream(x)
        gate, feat = self.shape_stream(x, res2, res3, res4, grad)
        out = self.feature_fusion(res1, res4, feat)
        seg = F.interpolate(self.seg(out), grad.size()[-2:], mode='bilinear', align_corners=False)
        # [B, N, H, W], [B, 1, H, W]
        return seg, gate
