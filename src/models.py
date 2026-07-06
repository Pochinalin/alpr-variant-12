# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = [
    'unet',
    'resnet_fpn',
    'deeplabv3',
    'pspnet',
    'segnet'
]

def model_display_names():
    return {
        'unet': 'U-Net',
        'resnet_fpn': 'ResNet50 + FPN',
        'deeplabv3': 'DeepLabV3',
        'pspnet': 'PSPNet',
        'segnet': 'SegNet'
    }

# ==================== 1. U-NET ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        self.final = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx](x)
        
        return torch.sigmoid(self.final(x))

# ==================== 2. RESNET50 + FPN ====================
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet50(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.out_channels = 2048
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
        
        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return [c2, c3, c4, c5]

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
    
    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        
        outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        x = outputs[0]
        for i in range(1, len(outputs)):
            x = F.interpolate(x, size=outputs[i].shape[2:], mode='nearest')
            x = x + outputs[i]
        
        return x

class ResNetFPN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = ResNet50(in_channels)
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)
        self.final = nn.Conv2d(256, out_channels, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        x = self.fpn(features)
        return torch.sigmoid(self.final(x))

# ==================== 3. DEEPLABV3 ====================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 6, 6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 12, 12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, 18, 18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_global = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.final = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        size = x.shape[2:]
        
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(x)))
        out3 = self.relu(self.bn3(self.conv3(x)))
        out4 = self.relu(self.bn4(self.conv4(x)))
        
        out5 = self.global_avg(x)
        out5 = self.relu(self.bn_global(self.conv_global(out5)))
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.relu(self.bn_final(self.final(out)))
        
        return out

class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.aspp = ASPP(512, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

# ==================== 4. PSPNET ====================
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ))
        
        self.final = nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.shape[2:]
        
        outputs = [x]
        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
            outputs.append(out)
        
        out = torch.cat(outputs, dim=1)
        out = self.relu(self.bn(self.final(out)))
        
        return out

class PSPNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.ppm = PyramidPooling(512, 512)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.ppm(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

# ==================== 5. SEGNET ====================
class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )
    
    def forward(self, x):
        # Encoder
        x = self.enc_conv1(x)
        x, idx1 = self.pool1(x)
        
        x = self.enc_conv2(x)
        x, idx2 = self.pool2(x)
        
        x = self.enc_conv3(x)
        x, idx3 = self.pool3(x)
        
        x = self.enc_conv4(x)
        x, idx4 = self.pool4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.unpool4(x, idx4)
        x = self.dec_conv4(x)
        
        x = self.unpool3(x, idx3)
        x = self.dec_conv3(x)
        
        x = self.unpool2(x, idx2)
        x = self.dec_conv2(x)
        
        x = self.unpool1(x, idx1)
        x = self.dec_conv1(x)
        
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return torch.sigmoid(x)

# ==================== ФАБРИКА МОДЕЛЕЙ ====================
def create_model(model_name, in_channels=1):
    """Создание модели по имени"""
    if model_name == 'unet':
        return UNet(in_channels=in_channels)
    elif model_name == 'resnet_fpn':
        return ResNetFPN(in_channels=in_channels)
    elif model_name == 'deeplabv3':
        return DeepLabV3(in_channels=in_channels)
    elif model_name == 'pspnet':
        return PSPNet(in_channels=in_channels)
    elif model_name == 'segnet':
        return SegNet(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")