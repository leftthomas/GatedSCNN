import argparse
import glob
import math
import os
import time

import pandas as pd
import torch
from cityscapesscripts.helpers.labels import trainId2label
from thop import profile, clever_format
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset import palette, Cityscapes
from model import GatedSCNN


# train or val for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_time, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, name in data_bar:
            data, target = data.cuda(), target.cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            out = net(data)
            prediction = torch.argmax(out, dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(prediction == target).item() / target.numel() * data.size(0)

            if not is_train and epoch % val_step == 0:
                # revert train id to regular id
                for key in trainId2label.keys():
                    prediction[prediction == key] = trainId2label[key].id
                # save pred images
                for pred_tensor, pred_name in zip(prediction, name):
                    pred_img = ToPILImage()(pred_tensor.unsqueeze(dim=0).byte().cpu())
                    pred_img.putpalette(palette)
                    pred_img.save('results/{}'.format(pred_name.replace('leftImg8bit', 'color')))
                # eval predicted results
                os.system('csEvalPixelLevelSemanticLabeling')

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} mPA: {:.2f}% FPS: {:.0f}'
                                     .format('Train' if is_train else 'Val', epoch, epochs, total_loss / total_num,
                                             total_correct / total_num * 100, total_num / total_time))

    return total_loss / total_num, total_correct / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gated-SCNN')
    parser.add_argument('--data_path', default='/home/data/cityscapes', type=str,
                        help='Data path for cityscapes dataset')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnet101'],
                        help='Backbone type')
    parser.add_argument('--crop_h', default=800, type=int, help='Crop height for training images')
    parser.add_argument('--crop_w', default=800, type=int, help='Crop width for training images')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of data for each batch to train')
    parser.add_argument('--val_step', default=5, type=int, help='Number of steps to val predicted results')
    parser.add_argument('--epochs', default=230, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_path', default='results', type=str, help='Save path for predicted results')

    # args parse
    args = parser.parse_args()
    data_path, backbone_type, crop_h, crop_w = args.data_path, args.backbone_type, args.crop_h, args.crop_w
    batch_size, val_step, epochs, save_path = args.batch_size, args.val_step, args.epochs, args.save_path
    if not os.path.exists('results'):
        os.mkdir('results')
    # config the environment variable
    os.environ['CITYSCAPES_DATASET'] = data_path
    os.environ['CITYSCAPES_RESULTS'] = save_path
    search_path = os.path.join(os.getenv('CITYSCAPES_DATASET'), 'gtFine', '*', '*', '*labelTrainIds.png')
    if not glob.glob(search_path):
        os.system('csCreateTrainIdLabelImgs')

    # dataset, model setup and optimizer config
    train_data = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data = Cityscapes(root=data_path, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    model = GatedSCNN(backbone_type=backbone_type, num_classes=19).cuda()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda eiter: math.pow(1 - eiter / epochs, 1.0))

    # model profile and loss definition
    flops, params = profile(model, inputs=(torch.randn(1, 3, crop_h, crop_w).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    loss_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # training loop
    results = {'train_loss': [], 'val_loss': [], 'train_mPA': [], 'val_mPA': []}
    best_mPA = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_mPA = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_mPA'].append(train_mPA)
        val_loss, val_mPA = train_val(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_mPA'].append(val_mPA)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}_{}_statistics.csv'.format(crop_h, crop_w), index_label='epoch')
        if val_mPA > best_mPA:
            best_mPA = val_mPA
            torch.save(model.state_dict(), '{}_{}_model.pth'.format(crop_h, crop_w))
