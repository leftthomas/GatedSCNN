import glob

from cityscapesscripts.helpers.labels import trainId2label
from torchvision import transforms
from tqdm import tqdm

city_mean, city_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])

palette = []
for key in sorted(trainId2label.keys()):
    if key != -1 and key != 255:
        palette += list(trainId2label[key].color)


def compute_metrics(pred_root, target_root, ignore_label=255):
    preds = glob.glob(pred_root)
    targets = glob.glob(target_root)
    preds.sort()
    targets.sort()
    for pred, target in tqdm(zip(preds, targets), desc='calculating the metrics'):
        pa = None
    return pa
