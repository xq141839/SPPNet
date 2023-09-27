# SPPNet: A Single-Point Prompt Network for Nuclei Image Segmentation (Boost SAM)

## News
2023.07.14: The SPPNet model and training code have been submitted. The paper will be updated later.

2023.08.24: The paper has been accepted by [MICCAI-MLMI 2023](https://sites.google.com/view/mlmi2023). The preprint has been available at [arXiv](https://arxiv.org/abs/2308.12231).

2023.09.27: Release a New Beta version for users who want to fine-tune the SAM pre-trained image encoder. We add the adapter based on [Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter).

## Requirements
1. pytorch==1.10.0
2. pytorch-lightning==1.1.0
3. albumentations==0.3.2
4. seaborn
5. sklearn

## Environment
NVIDIA RTX2080Ti Tensor Core GPU, 4-core CPU, and 28GB RAM

## Evaluation on MoNuSeg-2018

| Method| mIoU(%) | DSC(%) | Params(M) | FLOPs | FPS |
|  ----  |  ----  | ----  | ----  | ----  | ----  |
| SAM (Fine-tuned) | 60.18±8.15 | 74.76±7.00 | 635.93 | 2736.63 | 1.39| 
| SPPNet  | 66.43±4.32 | 79.77±3.11 | 9.79 | 39.90 | 22.61 | 

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

