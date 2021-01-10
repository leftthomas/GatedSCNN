# YOLOv4-p7

This is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)"
using PyTorch framwork.

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>
S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput | | :-- | :-:
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | | **YOLOv4-P7** | 1536 | **55.4%** | **73.3%** | **60.7%** | **38.1%** | **
59.5%** | **67.4%** | 15 *fps* | | **YOLOv4-P7** | TTA | **55.8%** | **73.2%** | **61.2%** | **38.8%** | **60.1%** | **
68.2%** | - | | | | | | | | |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>
S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | weights | | :-- | :-: | :-: | :-:
| :-: | :-: | :-: | :-: | :-: | | **YOLOv4-P7** | 1536 | **55.0%** | **72.9%** | **60.2%** | **39.8%** | **59.9%** | **
68.4%** | [`yolov4-p7.pt`](https://drive.google.com/file/d/18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3/view?usp=sharing)  | | **
YOLOv4-P7** | TTA | **55.5%** | **72.9%** | **60.8%** | **41.1%** | **60.3%** | **68.9%** | - | | | | | | | | | |

## Installation

```
pip install opencv-python thop pycocotools
```

## Testing

```
# download yolov4-p7.pt and put them in weights/ folder.
python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data config/coco.yaml --weights weights/yolov4-p7.pt
```

You will get following results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.55046
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.72925
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.60224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.39836
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.59854
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.68405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.40256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.66929
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.72943
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.59943
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.76873
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84460
```

## Training

We use multiple GPUs for training, YOLOv4-P7 use input resolution 1536 for training.

```
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 1536 1536 --data config/coco.yaml --cfg config/yolov4-p7.yaml --weights '' --sync-bn --device 0,1,2,3 --name yolov4-p7
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 1536 1536 --data config/coco.yaml --cfg config/yolov4-p7.yaml --weights 'runs/exp0_yolov4-p7/weights/last_298.pt' --sync-bn --device 0,1,2,3 --name yolov4-p7-tune --hyp 'config/hyp.finetune.yaml' --epochs 450 --resume
```

If your training process stucks, it due to bugs of the python. Just `Ctrl+C` to stop training and resume training by:

```
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 1536 1536 --data config/coco.yaml --cfg config/yolov4-p7.yaml --weights 'runs/exp0_yolov4-p7/weights/last.pt' --sync-bn --device 0,1,2,3 --name yolov4-p7 --resume
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
