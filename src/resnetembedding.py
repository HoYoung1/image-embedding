import torch
import torchvision.models as models


class ResnetEmbedder:

    def __init__(self, device=None):
        # Set up default device gpu or cpu
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=True)
        # Copy model to device
        self.model.to(device=self.device)

    def __call__(self, images_tensor):
        self.model.eval()
        with torch.no_grad():
            # Copy batch to device either CPU to GPU
            input = images_tensor.to(device=self.device)
            result = self.model(input)
            return result
