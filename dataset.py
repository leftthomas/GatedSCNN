import glob
import os
import random
import sys
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data import Dataset

from utils import city_mean, city_std

grad_progress = 0
boundary_progress = 0


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
        search_boundaries = os.path.join(root, 'gtFine', split, '*', '*boundary.png')
        self.images = glob.glob(search_images)
        self.labels = glob.glob(search_labels)
        self.grads = glob.glob(search_grads)
        self.boundaries = glob.glob(search_boundaries)
        self.images.sort()
        self.labels.sort()
        self.grads.sort()
        self.boundaries.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        grad_path = self.grads[index]
        boundary_path = self.boundaries[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        grad = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
        boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        name = image_path.split('/')[-1]

        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 1.0, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            grad = cv2.resize(grad, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        grad = np.asarray(grad, np.float32)
        boundary = np.asarray(boundary, np.float32)
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
                boundary_pad = cv2.copyMakeBorder(boundary, 0, pad_h, 0,
                                                  pad_w, cv2.BORDER_CONSTANT,
                                                  value=(0,))
            else:
                img_pad, label_pad, grad_pad, boundary_pad = image, label, grad, boundary

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            grad = grad_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            boundary = boundary_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.long)

        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            grad = grad[:, ::flip]
            boundary = boundary[:, ::flip]

        return image.copy(), label.copy(), np.expand_dims(grad, axis=0).copy(), boundary.copy(), name


def generate_grad(image_name, total_num):
    # create the output filename
    dst = image_name.replace('/leftImg8bit/', '/gtFine/')
    dst = dst.replace('_leftImg8bit', '_gtFine_grad')
    # do the conversion
    grad_image = cv2.Canny(cv2.imread(image_name), 10, 100)
    cv2.imwrite(dst, grad_image)
    global grad_progress
    grad_progress += 1
    print("\rProgress: {:>3} %".format(grad_progress * 100 / total_num), end=' ')
    sys.stdout.flush()


def generate_boundary(image_name, num_classes, ignore_label, total_num):
    # create the output filename
    dst = image_name.replace('_labelTrainIds', '_boundary')
    # do the conversion
    semantic_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    onehot_image = np.array([semantic_image == i for i in range(num_classes)]).astype(np.uint8)
    # change the ignored label to 0
    onehot_image[onehot_image == ignore_label] = 0
    boundary_image = np.zeros(onehot_image.shape[1:])
    # for boundary conditions
    onehot_image = np.pad(onehot_image, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(onehot_image[i, :]) + distance_transform_edt(1.0 - onehot_image[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > 2] = 0
        boundary_image += dist
    boundary_image = (boundary_image > 0).astype(np.uint8)
    cv2.imwrite(dst, boundary_image)
    global boundary_progress
    boundary_progress += 1
    print("\rProgress: {:>3} %".format(boundary_progress * 100 / total_num), end=' ')
    sys.stdout.flush()


def creat_dataset(root, num_classes=19, ignore_label=255):
    search_path = os.path.join(root, 'leftImg8bit', '*', '*', '*leftImg8bit.png')
    if not glob.glob(search_path):
        if not os.path.exists(root):
            os.makedirs(root)
        # download dataset
        os.system('csDownload -r -d {} gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip'.format(root))
        os.system("unzip -o {}/'*.zip' -d {}".format(root, root))
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
        # use multiprocessing to generate grad images
        print('\nGenerating {} grad images'.format(len(files)))
        print("Progress: {:>3} %".format(grad_progress * 100 / len(files)), end=' ')
        pool = ThreadPool()
        pool.map(partial(generate_grad, total_num=len(files)), files)
        pool.close()
        pool.join()

    search_path = os.path.join(root, 'gtFine', '*', '*', '*boundary.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'gtFine', '*', '*', '*labelTrainIds.png')
        files = glob.glob(search_path)
        files.sort()
        # use multiprocessing to generate boundary images
        print('\nGenerating {} boundary images'.format(len(files)))
        print("Progress: {:>3} %".format(boundary_progress * 100 / len(files)), end=' ')
        pool = ThreadPool()
        pool.map(partial(generate_boundary, num_classes=num_classes, ignore_label=ignore_label, total_num=len(files)),
                 files)
        pool.close()
        pool.join()
