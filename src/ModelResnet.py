from torch import nn
from torchvision import models


class ModelResnet(nn.Module):

    def __init__(self, n_classes, drop_out=0.5):
        super().__init__()
        self.drop_out = drop_out
        self.model = models.resnet50(pretrained=False)
        self.classes = n_classes
        # Change the final layer so that the number of classes
        # Use print final layer to figure out the input size to the final layer
        # print(self.model.fc)
        fc_input_size = 2048
        self.model.fc = nn.Sequential(nn.Dropout(p=self.drop_out), nn.Linear(fc_input_size, self.classes))

    def forward(self, input):
        fc_out = self.model(input)
        return fc_out
