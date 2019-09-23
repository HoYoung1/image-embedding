import glob
import tempfile

import torch
from PIL import Image
from torch import load
from torchvision import transforms


def input_fn(request_body, request_content_type):
    """An input_fn that processes the request body to a tensor"""
    if request_content_type == 'application/binary':

        with tempfile.NamedTemporaryFile("w+b") as f:
            f.write(request_body)
            f.seek(0)
            result = _pre_process_image(f)

        return result
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise "Unsupported content type {}".format(request_content_type)


def model_fn(model_dir):
    model_file = _find_artifact("{}/*.pt".format(model_dir))
    model = load(model_file)
    return model


def predict_fn(input_data, model):
    """Predict using input and model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))


def output_fn(prediction, content_type):
    """Return prediction"""
    return prediction


def _find_artifact(pattern):
    matching = glob.glob(pattern)
    assert len(matching) == 1, "Expected exactly one in {}, but found {}".format(pattern,
                                                                                 len(matching))
    matched_file = matching[0]
    return matched_file


def _pre_process_image(image_fp):
    # pre-process data
    image = Image.open(image_fp)
    # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(image)
    # Add batch [N, C, H, W]
    img_tensor = img.unsqueeze(0)
    return img_tensor
