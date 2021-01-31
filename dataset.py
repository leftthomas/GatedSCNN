import glob
import os
import random

import cv2
import numpy as np
import torch
from cityscapesscripts.helpers.labels import trainId2label
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

city_mean, city_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])

palette = []
for key in sorted(trainId2label.keys()):
    if key != -1 and key != 255:
        palette += list(trainId2label[key].color)


class Cityscapes(Dataset):
    def __init__(self, root, split='train', crop_size=None, mean=city_mean, std=city_std, ignore_label=255):

        self.split = split
        if split == 'train':
            assert crop_size is not None
            self.crop_h, self.crop_w = crop_size
        self.mean, self.std = mean, std
        self.ignore_label = ignore_label

        search_images = os.path.join(root, 'leftImg8bit', split, '*', '*leftImg8bit.png')
        search_labels = os.path.join(root, 'gtFine', split, '*', '*labelTrainIds.png')
        search_grads = os.path.join(root, 'gtFine', split, '*', '*grad.png')
        self.images = glob.glob(search_images)
        self.labels = glob.glob(search_labels)
        self.grads = glob.glob(search_grads)
        self.images.sort()
        self.labels.sort()
        self.grads.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        grad_path = self.grads[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        grad = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
        name = image_path.split('/')[-1]

        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 1.0, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            grad = cv2.resize(grad, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        grad = np.asarray(grad, np.float32)
        # change to RGB
        image = image[:, :, ::-1]
        # normalization
        image /= 255.0
        image -= self.mean
        image /= self.std
        grad /= 255.0

        # random crop
        if self.split == 'train':
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
                grad_pad = cv2.copyMakeBorder(grad, 0, pad_h, 0,
                                              pad_w, cv2.BORDER_CONSTANT,
                                              value=(0,))
            else:
                img_pad, label_pad, grad_pad = image, label, grad

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            grad = grad_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.long)

        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            grad = grad[:, ::flip]

        return image.copy(), label.copy(), np.expand_dims(grad, axis=0).copy(), name


def creat_dataset(root):
    search_path = os.path.join(root, 'leftImg8bit', '*', '*', '*leftImg8bit.png')
    if not glob.glob(search_path):
        if not os.path.exists(root):
            os.mkdir(root)
        # download dataset
        os.system('csDownload -d {} gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip'.format(root))
        os.system("unzip {}/'*.zip' -d {}".format(root, root))
    search_path = os.path.join(root, 'gtFine', '*', '*', '*labelTrainIds.png')
    if not glob.glob(search_path):
        # config the environment variable
        os.environ['CITYSCAPES_DATASET'] = root
        # generate pixel labels
        os.system('csCreateTrainIdLabelImgs')
    search_path = os.path.join(root, 'gtFine', '*', '*', '*grad.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'leftImg8bit', '*', '*', '*leftImg8bit.png')
        files = glob.glob(search_path)
        files.sort()
        # generate grad images
        for f in tqdm(files, desc='generating grad images'):
            # create the output filename
            dst = f.replace('/leftImg8bit/', '/gtFine/')
            dst = dst.replace('_leftImg8bit', '_gtFine_grad')
            # do the conversion
            grad_image = cv2.Canny(cv2.imread(f), 10, 100)
            cv2.imwrite(dst, grad_image)


def compute_metric(output, target):
    pa = torch.sum(torch.eq(output, target)) / target.numel()
    return pa
