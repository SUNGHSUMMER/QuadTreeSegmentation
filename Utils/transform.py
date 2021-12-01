from torchvision import transforms
from PIL import Image
import numpy as np
import torch

transformer = transforms.Compose([
    transforms.ToTensor(),
])


def resize(images, shape, label=False):
    """
    resize PIL images
    shape: (w, h)
    """
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized


def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    return target


def masks_transform(masks, numpy=False):
    """
    masks: list of PIL images
    """
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()


def images_transform(images):
    """
    images: list of PIL images
    """
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs
