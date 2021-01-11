# YOLOv4-p7

This is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)"
using PyTorch framwork.

## Installation

```
pip install opencv-python thop
```

## Testing

```
python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data config/data.yaml --weights best.pt
```

## Training

We use multiple GPUs for training, YOLOv4-P7 use input resolution 1536 for training.

```
python train.py --batch-size 64 --img 640 640 --data config/data.yaml --cfg config/model.yaml --weights '' --sync-bn --device 0,1,2,3 --name yolov4-p7
python train.py --batch-size 64 --img 640 640 --data config/data.yaml --cfg config/model.yaml --weights 'runs/exp0/weights/last_298.pt' --sync-bn --device 0,1,2,3 --name model-tune --hyp 'config/finetune.yaml' --epochs 450 --resume
```

If your training process stucks, it due to bugs of the python. Just `Ctrl+C` to stop training and resume training by:

```
python train.py --batch-size 64 --img 640 640 --data config/data.yaml --cfg config/model.yaml --weights 'runs/exp0/weights/last.pt' --sync-bn --device 0,1,2,3 --resume
```

## Citation

```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```
