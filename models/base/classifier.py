'''
Function:
    classification layer for all supported CNN-based backbones
Author:
    Changtao Miao
'''
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, inplanes, num_classes=2):
        super(Classifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
