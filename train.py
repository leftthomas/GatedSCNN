import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from cityscapesscripts.helpers.labels import trainId2label
from torch import nn
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset import creat_dataset, Cityscapes
from model import GatedSCNN
from utils import get_palette, compute_metrics, BoundaryBCELoss, DualTaskLoss

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train or val or test for one epoch
def for_loop(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_time, total_num, preds, targets = 0.0, 0.0, 0, [], []
    data_bar = tqdm(data_loader, dynamic_ncols=True)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, grad, boundary, name in data_bar:
            data, target, grad, boundary = data.cuda(), target.cuda(), grad.cuda(), boundary.cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            seg, edge = net(data, grad)
            prediction = torch.argmax(seg.detach(), dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            semantic_loss = semantic_criterion(seg, target)
            edge_loss = edge_criterion(edge, target, boundary)
            task_loss = task_criterion(seg, edge, target)
            loss = semantic_loss + 20 * edge_loss + task_loss

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            preds.append(prediction.cpu())
            targets.append(target.cpu())

            if not is_train:
                # revert train id to regular id
                for key in sorted(trainId2label.keys(), reverse=True):
                    prediction[prediction == key] = trainId2label[key].id
                # save pred images
                save_root = '{}/{}'.format(save_path, data_loader.dataset.split)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                for pred_tensor, pred_name in zip(prediction, name):
                    pred_img = ToPILImage()(pred_tensor.unsqueeze(dim=0).byte().cpu())
                    pred_img.putpalette(get_palette())
                    pred_name = pred_name.replace('leftImg8bit', 'color')
                    path = '{}/{}'.format(save_root, pred_name)
                    pred_img.save(path)
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} FPS: {:.0f}'
                                     .format(data_loader.dataset.split.capitalize(), epoch, epochs,
                                             total_loss / total_num, total_num / total_time))
        # compute metrics
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        pa, mpa, class_iou, category_iou = compute_metrics(preds, targets)
        print('{} Epoch: [{}/{}] PA: {:.2f}% mPA: {:.2f}% Class_mIOU: {:.2f}% Category_mIOU: {:.2f}%'
              .format(data_loader.dataset.split.capitalize(), epoch, epochs,
                      pa * 100, mpa * 100, class_iou * 100, category_iou * 100))
    return total_loss / total_num, pa * 100, mpa * 100, class_iou * 100, category_iou * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gated-SCNN')
    parser.add_argument('--data_path', default='data', type=str, help='Data path for cityscapes dataset')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnet101'],
                        help='Backbone type')
    parser.add_argument('--crop_h', default=800, type=int, help='Crop height for training images')
    parser.add_argument('--crop_w', default=800, type=int, help='Crop width for training images')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of data for each batch to train')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_path', default='results', type=str, help='Save path for results')

    # args parse
    args = parser.parse_args()
    data_path, backbone_type, crop_h, crop_w = args.data_path, args.backbone_type, args.crop_h, args.crop_w
    batch_size, epochs, save_path = args.batch_size, args.epochs, args.save_path

    # dataset, model setup, optimizer config and loss definition
    creat_dataset(data_path)
    train_data = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data = Cityscapes(root=data_path, split='val')
    test_data = Cityscapes(root=data_path, split='test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    model = GatedSCNN(backbone_type=backbone_type, num_classes=19).cuda()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda eiter: math.pow(1 - eiter / epochs, 1.0))
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=255)
    edge_criterion = BoundaryBCELoss(ignore_index=255)
    task_criterion = DualTaskLoss(threshold=0.8, ignore_index=255)

    results = {'train_loss': [], 'val_loss': [], 'train_PA': [], 'val_PA': [], 'train_mPA': [], 'val_mPA': [],
               'train_class_mIOU': [], 'val_class_mIOU': [], 'train_category_mIOU': [], 'val_category_mIOU': []}
    best_mIOU = 0.0
    # train/val/test loop
    for epoch in range(1, epochs + 1):
        train_loss, train_PA, train_mPA, train_class_mIOU, train_category_mIOU = for_loop(model, train_loader,
                                                                                          optimizer)
        results['train_loss'].append(train_loss)
        results['train_PA'].append(train_PA)
        results['train_mPA'].append(train_mPA)
        results['train_class_mIOU'].append(train_class_mIOU)
        results['train_category_mIOU'].append(train_category_mIOU)
        val_loss, val_PA, val_mPA, val_class_mIOU, val_category_mIOU = for_loop(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_PA'].append(val_PA)
        results['val_mPA'].append(val_mPA)
        results['val_class_mIOU'].append(val_class_mIOU)
        results['val_category_mIOU'].append(val_category_mIOU)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_{}_{}_statistics.csv'.format(save_path, backbone_type, crop_h, crop_w),
                          index_label='epoch')
        if val_class_mIOU > best_mIOU:
            best_mIOU = val_class_mIOU
            # use best model to obtain the test results
            for_loop(model, test_loader, None)
            torch.save(model.state_dict(), '{}/{}_{}_{}_model.pth'.format(save_path, backbone_type, crop_h, crop_w))
