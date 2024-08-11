class RegNetModel(ImageClassificationBase):
    def __init__(self, num_classes=5, freeze_backbone = False):
        super().__init__()
        self.backbone = timm.create_model('regnetx_160.tv2_in1k', pretrained=True)
        #self.backbone = timm.create_model('resnet18d.ra2_in1k', pretrained=True)
        #self.backbone = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        # Modify the classifier to match the number of output classes
        in_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity() #must check if last layers is really fc by print(model)
        self.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            self.backbone.eval()
        #print(self)
    def forward(self, xb):
        feature_map = self.backbone(xb)
        output = self.fc(feature_map)
        return output