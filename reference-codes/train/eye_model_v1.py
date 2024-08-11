import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Eye_Net(nn.Module):
    def __init__(self, model_key, num_classes=3, in_channel=3):
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
