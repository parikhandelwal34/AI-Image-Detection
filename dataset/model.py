import torchvision.models as models
import torch.nn as nn

def get_model():
    model = models.resnet50(pretrained=True)

    # 6-channel input
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model