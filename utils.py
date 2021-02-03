import torch
import torch.nn.functional as F
from cityscapesscripts.helpers.labels import trainId2label
from torch import nn
from torchvision import transforms
from tqdm import tqdm

city_mean, city_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])


def get_palette():
    palette = []
    for key in sorted(trainId2label.keys()):
        if key != -1 and key != 255:
            palette += list(trainId2label[key].color)
    return palette


class BoundaryBCELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, edge, target, boundary):
        edge = edge.squeeze(dim=1)
        mask = target != self.ignore_index
        pos_mask = (boundary == 1.0) & mask
        neg_mask = (boundary == 0.0) & mask
        num = torch.clamp(mask.sum(), min=1)
        pos_weight = neg_mask.sum() / num
        neg_weight = pos_mask.sum() / num

        weight = torch.zeros_like(boundary)
        weight[pos_mask] = pos_weight
        weight[neg_mask] = neg_weight
        loss = F.binary_cross_entropy(edge, boundary, weight, reduction='sum') / num
        return loss


class DualTaskLoss(nn.Module):
    def __init__(self, threshold=0.8, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index

    def forward(self, seg, edge, target):
        edge = edge.squeeze(dim=1)
        logit = F.cross_entropy(seg, target, ignore_index=self.ignore_index, reduction='none')
        mask = target != self.ignore_index
        num = torch.clamp(((edge > self.threshold) & mask).sum(), min=1)
        loss = (logit[edge > self.threshold].sum()) / num
        return loss


def compute_metrics(preds, targets, ignore_label=255, num_classes=19, num_category=7):
    correct_pixel, total_pixel = torch.zeros(1, device=preds.device), torch.zeros(1, device=preds.device)
    class_tt = torch.zeros(num_classes, device=preds.device)
    class_tf, class_ft = torch.zeros(num_classes, device=preds.device), torch.zeros(num_classes, device=preds.device)
    category_tt = torch.zeros(num_category, device=preds.device)
    category_tf, category_ft = torch.zeros(num_category, device=preds.device), torch.zeros(num_category,
                                                                                           device=preds.device)
    for pred, target in tqdm(zip(preds, targets), desc='calculating the metrics', total=len(preds), dynamic_ncols=True):
        mask = target != ignore_label
        correct_pixel += torch.eq(pred, target)[mask].sum()
        total_pixel += mask.sum()
        for label in range(num_classes):
            class_tf_mask = (target == label) & mask
            class_ft_mask = (pred == label) & mask
            class_tt[label] += (class_tf_mask & class_ft_mask).sum()
            class_tf[label] += class_tf_mask.sum()
            class_ft[label] += class_ft_mask.sum()
        # revert train id to category id
        for key in sorted(trainId2label.keys(), reverse=True):
            # avoid overwrite label
            pred[pred == key] = trainId2label[key].categoryId + 1000
            target[target == key] = trainId2label[key].categoryId + 1000
        # back true label, and use 0-index label
        pred -= 1001
        target -= 1001
        for label in range(num_category):
            category_tf_mask = (target == label) & mask
            category_ft_mask = (pred == label) & mask
            category_tt[label] += (category_tf_mask & category_ft_mask).sum()
            category_tf[label] += category_tf_mask.sum()
            category_ft[label] += category_ft_mask.sum()
    pa = correct_pixel / torch.clamp(total_pixel, min=1)
    mpa = (class_tt / torch.clamp(class_tf, min=1)).mean()
    class_iou = (class_tt / torch.clamp(class_tf + class_ft - class_tt, min=1)).mean()
    category_iou = (category_tt / torch.clamp(category_tf + category_ft - category_tt, min=1)).mean()
    return pa.item(), mpa.item(), class_iou.item(), category_iou.item()
