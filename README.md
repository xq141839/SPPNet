# SPPNet: A Single-Point Prompt Network for Nuclei Image Segmentation (Boost SAM)

## News
2023.07.14: The SPPNet model and training code have been submitted. The paper will be updated later.

## Requirements
1. pytorch==1.10.0
2. pytorch-lightning==1.1.0
3. albumentations==0.3.2
4. seaborn
5. sklearn

## Environment
NVIDIA RTX2080Ti Tensor Core GPU, 4-core CPU, and 28GB RAM

## Dataset
To apply the model on a custom dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.npy
                ├── image_2.npy
                ├── image_n.npy
```
## Train
```
python train.py --dataset your/data/path --jsonfile your/json/path --loss dice --batch 16 --lr 0.001 --epoch 50 
```
## Evaluation
```
python eval.py --dataset your/data/path --jsonfile your/json/path --model save_models/model_best.pth --debug True
```
## Acknowledgement
The codes are modified from [SAM](https://github.com/facebookresearch/segment-anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

