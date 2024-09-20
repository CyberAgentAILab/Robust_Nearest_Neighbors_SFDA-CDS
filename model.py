import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

class Resnet(nn.Module):
    def __init__(self, arch, num_class, pretrained=False):
        super(Resnet, self).__init__()
        self.bottleneck_dim = 256

        if arch == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == 'resnet101':
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif arch == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        if pretrained:
            self.model.load_state_dict(torch.load('logs/' + pretrained + '/pretrained_gen-model.tar'))
            
        self.model.fc = nn.Linear(self.model.fc.in_features, self.bottleneck_dim)
        bn = nn.BatchNorm1d(self.bottleneck_dim)
        self.encoder = nn.Sequential(self.model, bn)

        self.fc = nn.Linear(self.bottleneck_dim, num_class)

        self.fc = nn.utils.weight_norm(self.fc, dim=0)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)

        logits = self.fc(features)
     
        return features, logits
