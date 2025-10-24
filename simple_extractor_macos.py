#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
          'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    }
}


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette



def main(input_dir="./input", output_dir="./result", model_path="checkpoints/final.pth"):
    device = torch.device("cpu")
    settings = dataset_settings['lip']
    input_size = settings['input_size']
    num_classes = settings['num_classes']
    print(f"📊 Dataset: LIP | Classes: {num_classes}")

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SimpleFolderDataset(root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    os.makedirs(output_dir, exist_ok=True)
    palette = get_palette(num_classes)

    # 3. Inference
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            image = image.to(device)
            outputs = model(image)
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(outputs[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze().permute(1, 2, 0)  # CHW → HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            out_path = os.path.join(output_dir, img_name[:-4] + ".png")
            out_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            out_img.putpalette(palette)
            out_img.save(out_path)

    print("All results are saved in:", output_dir)


if __name__ == '__main__':
    main(input_dir="./input", output_dir="./result", model_path="checkpoints/final.pth")
