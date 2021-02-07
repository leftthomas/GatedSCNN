import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from model import GatedSCNN
from utils import transform, get_palette

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
    parser.add_argument('--data_path', default='data', type=str, help='Data path for cityscapes dataset')
    parser.add_argument('--model_weight', type=str, default='results/resnet50_512_512_model.pth',
                        help='Pretrained model weight')
    parser.add_argument('--input_pic', type=str, default='test/berlin/berlin_000000_000019_leftImg8bit.png',
                        help='Path to the input picture')
    # args parse
    args = parser.parse_args()
    data_path, model_weight, input_pic = args.data_path, args.model_weight, args.input_pic

    image = Image.open('{}/leftImg8bit/{}'.format(data_path, input_pic)).convert('RGB')
    image_height, image_width = image.height, image.width
    num_width = 2 if 'test' in input_pic else 3
    target = Image.new('RGB', (image_width * num_width, image_height))
    images = [image]

    image = transform(image).unsqueeze(dim=0).cuda()
    grad = cv2.Canny(cv2.imread('{}/leftImg8bit/{}'.format(data_path, input_pic)), 10, 100)
    grad = torch.from_numpy(np.expand_dims(np.asarray(grad, np.float32) / 255.0, axis=0).copy()).unsqueeze(dim=0).cuda()

    # model load
    model = GatedSCNN(model_weight.split('/')[-1].split('_')[0], num_classes=19)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))
    model = model.cuda()
    model.eval()

    # predict and save image
    with torch.no_grad():
        output, _ = model(image, grad)
        pred = torch.argmax(output, dim=1)
        pred_image = ToPILImage()(pred.byte().cpu())
        pred_image.putpalette(get_palette())
        if 'test' not in input_pic:
            gt_image = Image.open('{}/gtFine/{}'.format(data_path, input_pic.replace('leftImg8bit', 'gtFine_color')))
            images.append(gt_image)
        images.append(pred_image)
        # concat images
        for i in range(len(images)):
            left, top, right, bottom = image_width * i, 0, image_width * (i + 1), image_height
            target.paste(images[i], (left, top, right, bottom))
        target.save(os.path.split(input_pic)[-1].replace('leftImg8bit', 'result'))
