import argparse
import json

from utils.datasets import create_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', type=str, default='data/annotations/train_annos.json', help='labels file path')
    opt = parser.parse_args()
    with open(opt.label_file, 'r') as load_f:
        load_dict = json.load(load_f)
    path = 'data/train/labels'
    create_folder(path)
    for item in load_dict:
        file_name = '{}/{}'.format(path, item['name'].replace('jpg', 'txt'))
        file = open(file_name, mode='a')
        tx, ty, bx, by = item['bbox']
        w, h = item['image_width'], item['image_height']
        cx, cy = (tx + bx) / 2, (ty + by) / 2
        bw, bh = bx - tx, by - ty
        file.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(item['category'], cx / w, cy / h, bw / w, bh / h))
        file.close()
