import torch
from cityscapesscripts.helpers.labels import trainId2label
from torchvision import transforms

city_mean, city_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])

palette = []
for key in sorted(trainId2label.keys()):
    if key != -1 and key != 255:
        palette += list(trainId2label[key].color)


def compute_metric(output, target):
    pa = torch.sum(torch.eq(output, target)) / target.numel()
    return pa
