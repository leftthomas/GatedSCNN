import argparse
import glob
import math
import os
import sys
import time

import cv2
import pandas as pd
import torch
from cityscapesscripts.helpers.labels import trainId2label
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset import palette, Cityscapes
from model import GatedSCNN


# train or val or test for one epoch
def for_loop(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_pa, total_iou, total_time, total_num, data_bar = 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, grad, name in data_bar:
            data, target, grad = data.cuda(), target.cuda(), grad.cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            seg, edge = net(data, grad)
            prediction = torch.argmax(seg, dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            loss = loss_criterion(seg, edge, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            # compute PA
            total_pa += torch.sum(torch.eq(prediction, target)).item() / target.numel() * data.size(0)

            if data_loader.dataset.split == 'test':
                # revert train id to regular id
                for key in trainId2label.keys():
                    prediction[prediction == key] = trainId2label[key].id
                # save pred images
                for pred_tensor, pred_name in zip(prediction, name):
                    pred_img = ToPILImage()(pred_tensor.byte().cpu())
                    pred_img.putpalette(palette)
                    pred_name = pred_name.replace('leftImg8bit', 'color')
                    path = '{}/{}'.format(save_path, pred_name)
                    pred_img.save(path)
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} mPA: {:.2f}% mIOU: {:.2f}% FPS: {:.0f}'
                                     .format(data_loader.dataset.split.capitalize(), epoch, epochs,
                                             total_loss / total_num, total_pa / total_num * 100,
                                             total_iou / total_num * 100, total_num / total_time))
    return total_loss / total_num, total_pa / total_num * 100, total_iou / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gated-SCNN')
    parser.add_argument('--data_path', default='data', type=str, help='Data path for cityscapes dataset')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnet101'],
                        help='Backbone type')
    parser.add_argument('--crop_h', default=800, type=int, help='Crop height for training images')
    parser.add_argument('--crop_w', default=800, type=int, help='Crop width for training images')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of data for each batch to train')
    parser.add_argument('--epochs', default=230, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_path', default='results', type=str, help='Save path for results')

    # args parse
    args = parser.parse_args()
    data_path, backbone_type, crop_h, crop_w = args.data_path, args.backbone_type, args.crop_h, args.crop_w
    batch_size, epochs, save_path = args.batch_size, args.epochs, args.save_path
    search_path = os.path.join(data_path, 'gtFine', '*', '*', '*labelTrainIds.png')
    if not glob.glob(search_path):
        # config the environment variable, generate pixel labels
        os.environ['CITYSCAPES_DATASET'] = data_path
        os.system('csCreateTrainIdLabelImgs')
    # generate grad images
    search_path = os.path.join(data_path, 'gtFine', '*', '*', '*grad.png')
    if not glob.glob(search_path):
        search_path = os.path.join(data_path, 'leftImg8bit', '*', '*', '*leftImg8bit.png')
        files = glob.glob(search_path)
        files.sort()
        # a bit verbose
        print('Processing {} images to generate grad images'.format(len(files)))
        # iterate through files
        progress = 0
        print('Progress: {:>3} %'.format(progress * 100 / len(files)), end=' ')
        for f in files:
            # create the output filename
            dst = f.replace('/leftImg8bit/', '/gtFine/')
            dst = dst.replace('_leftImg8bit', '_gtFine_grad')
            # do the conversion
            try:
                grad_image = cv2.Canny(cv2.imread(f), 10, 100)
                cv2.imwrite(dst, grad_image)
            except:
                print("Failed to convert: {}".format(f))
                raise
            # status
            progress += 1
            print("\rProgress: {:>3} %".format(progress * 100 / len(files)), end=' ')
            sys.stdout.flush()

    # dataset, model setup, optimizer config and loss definition
    train_data = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data = Cityscapes(root=data_path, split='val')
    test_data = Cityscapes(root=data_path, split='test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    model = GatedSCNN(backbone_type=backbone_type, num_classes=19).cuda()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda eiter: math.pow(1 - eiter / epochs, 1.0))
    loss_criterion = nn.CrossEntropyLoss(ignore_index=255)

    results = {'train_loss': [], 'val_loss': [], 'train_mPA': [], 'val_mPA': [], 'train_mIOU': [], 'val_mIOU': []}
    best_mIOU = 0.0
    # train/val/test loop
    for epoch in range(1, epochs + 1):
        train_loss, train_mPA, train_mIOU = for_loop(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_mPA'].append(train_mPA)
        results['train_mIOU'].append(train_mIOU)
        val_loss, val_mPA, val_mIOU = for_loop(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_mPA'].append(val_mPA)
        results['val_mIOU'].append(val_mIOU)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_{}_{}_statistics.csv'.format(save_path, backbone_type, crop_h, crop_w),
                          index_label='epoch')
        if val_mIOU > best_mIOU:
            best_mIOU = val_mIOU
            # use best model to update the test results
            for_loop(model, test_loader, None)
            torch.save(model.state_dict(), '{}/{}_{}_{}_model.pth'.format(save_path, backbone_type, crop_h, crop_w))
