from torch import nn
from torchvision import models


class ModelResnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet_model = models.resnet50(pretrained=True)
        # Change the final layer so that the number of classes
        # Use print final layer to figure out the input size to the final layer
        # print(self.model.fc)
        # fc_input_size = 512  # 2048 is for resnet 50

    def forward(self, input):
        fc_out = self.resnet_model(input)
        return fc_out
