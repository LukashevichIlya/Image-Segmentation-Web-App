import io
from typing import Tuple

import torch
from PIL import Image
from fastapi import File
from torchvision import transforms


def get_segmentation_model() -> torch.nn.Module:
    """Download segmentation model from PyTorch Hub."""
    model = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True)
    model.eval()
    return model


def get_segmentation_map(model: torch.nn.Module,
                         image: bytes = File(...),
                         max_image_size=512) -> Image:
    """Get image segmentation map.

    Parameters
    ----------
    model
        Image segmentation model.
    image
        Image to be processed.
    max_image_size
        Maximum size for image dimensions.

    Returns
    -------
        Image representing segmentation map.

    """
    image = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = image.size
    scale = min(max_image_size / width, max_image_size / height)
    scaled_image = image.resize((int(image.width * scale),
                                 int(image.height * scale)))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    image_tensor = transform(scaled_image)
    image_batch = image_tensor.unsqueeze(0)

    with torch.no_grad():
        out = model(image_batch)["out"][0]
    predictions = out.argmax(0)

    res_segmentation_map = post_process_predictions(image.size, predictions)
    return res_segmentation_map


def post_process_predictions(image_size: Tuple[int, int], predictions: torch.Tensor) -> Image:
    """Assign color palette for each class of the predicted segmentation map.

    Parameters
    ----------
    image_size
        Size of the original image.
    predictions
        Predicted segmentation map.

    Returns
    -------
        Segmentation map with assigned colors to the predicted classes.

    References
    ----------
        The code for assigning color palette is taken from:
        https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

    """
    num_classes = 21
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    res_segmentation_map = Image.fromarray(predictions.byte().cpu().numpy()).resize(image_size)
    res_segmentation_map.putpalette(colors)
    return res_segmentation_map
