import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True, out_indices=(3, 4, 6)):
        super().__init__()
        
        model_constructor = getattr(models, model_name)
        model = model_constructor(weights="IMAGENET1K_V1" if pretrained else None)
        
        self.stem = model.features[:2]
        self.stage1 = model.features[2:4]
        self.stage2 = model.features[4:6]
        self.stage3 = model.features[6:]
        self.out_indices = out_indices

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f3
