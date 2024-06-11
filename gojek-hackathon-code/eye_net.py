import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Eye_Net(nn.Module):
    def __init__(self,model_key,in_channel=3,num_classes=3):
        super(Eye_Net, self).__init__()
        self.backbone = timm.create_model(model_key, in_chans=in_channel, pretrained=True, features_only=True, drop_rate=0.5)        
        self.dropout = nn.Dropout(0.5)
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(288,num_classes) 
        # 050: 288, 075: 432, 100: 576, large100: 960

    def forward(self, x):
        x = self.backbone(x)
        x = self.max_pool(x[-1])
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm


# class Eye_Net(nn.Module):
#     def __init__(self, in_channel, is_normalized = False, dropout = 0):
#         super(Eye_Net, self).__init__()
#         self.stem = nn.Conv2d(in_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.conv_head = nn.Conv2d(24, 256, kernel_size=(1, 1), stride=(1, 1))
#         self.backbone = timm.create_model('mobilenetv3_small_050', num_classes=3, pretrained=True, drop_rate=dropout)
#         # self.fc1 = nn.Linear(in_features=256, out_features=3, bias=True)
#         # self.fc2 = nn.Linear(in_features=64, out_features=3, bias=True)
#         # self.dropout = nn.Dropout(dropout)
        
#     def forward(self,x):
#         # x = self.stem(x)
#         # x = self.backbone.bn1(x)
#         # x = self.backbone.blocks[:3](x)
#         # x = self.backbone.global_pool(x)
#         # x = self.conv_head(x)
#         # x = self.backbone.act2(x)
#         # x = self.backbone.flatten(x)
#         # x = self.dropout(x)
#         # x = self.fc1(x)
#         # x = self.fc2(x)
#         x=self.backbone(x)
        
#         return x
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride=1, expand_ratio=1, kernel=3, padding=1):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         self.use_res_connect = self.stride == 1 and inp == oup

#         self.conv = nn.Sequential(
#             # pw
#             # nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#             # # nn.BatchNorm2d(inp * expand_ratio),
#             # nn.ReLU6(inplace=True),
#             # dw
#             nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, padding, groups=inp * expand_ratio, bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU6(inplace=True),
#             # pw-linear
#             nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         )

#     def forward(self, x):

#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)

# class R_Net(nn.Module):
#     def __init__(self,in_channel):
#         super(R_Net, self).__init__()
#         self.conv_r=InvertedResidual(64,64,kernel=7,stride=2)
#         self.fc1=nn.Sequential(
#             nn.Conv2d(64,32,1,1),
#             nn.BatchNorm2d(32),
#         )
#         self.fc2=nn.Sequential(
#             nn.Conv2d(32,3,1,1),
#             nn.BatchNorm2d(3),
#         )
        
#     def forward(self,input):
#         x=self.conv_r(input)
#         x_pool=F.adaptive_avg_pool2d(x,1)
#         x_fc1=self.fc1(x_pool)
#         x_fc2=self.fc2(x_fc1)
#         output=x_fc2.view(x_fc2.shape[0],-1)
#         return  output


# class Eye_Net(nn.Module):
#     def __init__(self, in_channel, is_normalized = False):
#         super(Eye_Net, self).__init__()
#         self.stem=nn.Sequential(
#             nn.Conv2d(in_channel, 32, kernel_size=3, stride=2,padding=1), # （256-3+2）//2+1=253//2+1=128
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1), # 127//2+1=64
#             nn.BatchNorm2d(64),
#         )
#         self.rnet=R_Net(in_channel = 64)
        
#     def forward(self,x):
#         stem_x=self.stem(x)
#         light_par=self.rnet(stem_x)
#         return light_par
    