import torch
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms


class ImageEmbedder:

    def __init__(self):

        self.model =  models.resnet18(pretrained=True)

    def __call__(self, image):
        self.model.eval()
        # torch image: C X H X W
        with torch.no_grad():
            image = Image.fromarray(image)
            min_img_size = 214  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
            transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])])
            img = transform_pipeline(image)

            img_tensor = img.unsqueeze(0)
            result = self.model(img_tensor)

        return result
