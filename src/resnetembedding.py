import torch
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms


class ResnetEmbedder:

    def __init__(self, min_img_size_h=214, min_img_size_w=214, device=None, ):
        self.min_img_size_h = min_img_size_h
        self.min_img_size_w = min_img_size_w

        # Set up default device
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet152(pretrained=True)
        self.model.to(device=self.device)

    def __call__(self, image):
        self.model.eval()

        with torch.no_grad():
            img_tensor = self._pre_process_image(image)
            img_tensor.to(device=self.device)
            result = self.model(img_tensor)

        return result

    def _pre_process_image(self, image):
        # pre-process data
        image = Image.fromarray(image)
        # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        transform_pipeline = transforms.Compose([transforms.Resize((self.min_img_size_h, self.min_img_size_w)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      # torch image: C X H X W
                                                                      std=[0.229, 0.224, 0.225])])
        img = transform_pipeline(image)
        # Add batch [N, C, H, W]
        img_tensor = img.unsqueeze(0)
        return img_tensor
