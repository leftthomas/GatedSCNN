# Gated-SCNN

A PyTorch implementation of Gated-SCNN based on ICCV 2019
paper [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](https://arxiv.org/abs/1907.05740).

![Network Architecture image from the paper](structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

- timm

```
pip install timm
```

- opencv

```
pip install opencv-python
```

- cityscapesScripts

```
pip install cityscapesscripts
```

## Expected dataset structure for Cityscapes:

```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json
      ...
    val/
  leftImg8bit/
    train/
    val/
```

## Usage

### Train model

```
python train.py --save_step 10 --epochs 175
optional arguments:
--data_path                   Data path for cityscapes dataset [default value is '/home/data/cityscapes']
--backbone_type               Backbone type [default value is 'resnet50'](choices=['resnet50', 'resnet101'])
--crop_h                      Crop height for training images [default value is 800]
--crop_w                      Crop width for training images [default value is 800]
--batch_size                  Number of data for each batch to train [default value is 16]
--val_step                    Number of steps to val predicted results [default value is 5]
--epochs                      Number of sweeps over the dataset to train [default value is 230]
--save_path                   Save path for predicted results [default value is 'results']
```

### Eval model

```
python viewer.py --model_weight 512_1024_model.pth
optional arguments:
--data_path                   Data path for cityscapes dataset [default value is '/home/data/cityscapes']
--model_weight                Pretrained model weight [default value is '1024_2048_model.pth']
--input_pic                   Path to the input picture [default value is 'test/berlin/berlin_000000_000019_leftImg8bit.png']
```

## Results

The experiment is conducted on one NVIDIA Tesla V100 (32G) GPU, and there are some difference between this
implementation and official implementation:

1. The scales of `Multi-Scale Training` are `(0.5, 1.0, 2.0)`;
2. No `dual task loss` used;
3. `Adam` optimizer with learning rate `1e-3` is used to train this model.

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Params (M)</th>
		<th>FLOPs (G)</th>
		<th>FPS</th>
		<th>Pixel Accuracy</th>
		<th>Class mIOU</th>
		<th>Category mIOU</th>
		<th>Download</th>
		<!-- TABLE BODY -->
		<tr>
			<td align="center">1.14</td>
			<td align="center">6.92</td>
			<td align="center">197</td>
			<td align="center">81.8</td>
			<td align="center">58.0</td>
			<td align="center">81.7</td>
			<td align="center"><a href="https://pan.baidu.com/s/1cmcAtDewYs2lWK7LaktofQ">model</a>&nbsp;|&nbsp;eg6a</td>
		</tr>
	</tbody>
</table>

The left is input image, the middle is ground truth segmentation, and the right is model's predicted segmentation.

![munster_000120_000019](result.png)
