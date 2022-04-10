import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential( #全连接层
            nn.Linear(512 * 7 * 7, 4096),#第一层全连接层
            nn.ReLU(True),
            nn.Dropout(), #减少过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):  #正向传播，x为输入的图像数据
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self): #定义初始化权重函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


def make_layers(cfg, batch_norm=False):#生成提取特征网络结构
    layers = [] #定义空列表，用来存放我们创建的每一层结构
    in_channels = 3 #输入rgb彩色图片，输入通道是3
    for v in cfg:  #遍历配置列表
        if v == 'M':  #最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)#创建一个卷积核
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] # 将定义好的卷积层和relu函数拼接在一起，并添加到layers列表
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # 把列表通过非关键字的形式传入进去

# 224,224,3 -> 224,224,64 -> 112,112,64 -> 112,112,128 -> 56,56,128 -> 56,56,256 -> 28,28,256 -> 28,28,512
# 14,14,512 -> 14,14,512 -> 7,7,512
cfgs = { # 定义字典文件，数字代表卷积核的个数，M代表池化层
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)

    if num_classes!=1000:
        model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model
