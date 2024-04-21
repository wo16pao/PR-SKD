# PR-SKD
This repo is the implementation of the paper [Enhancing deep feature representation in self-knowledge distillation via pyramid feature refinement](https://www.sciencedirect.com/science/article/abs/pii/S0167865523003616?via%3Dihub)

## Requirements
- Python3
- Pytorch (>1.4.0)
- torchvision
- numpy 

## Classification Training 
In this code, you can reproduce the experiment results of classification task in the paper.
The datasets are all open-sourced, so it is easy to download.
Example training settings are for ResNet18 on CIFAR-100.
Detailed hyperparameter settings are enumerated in the paper.

- Training with PR-SKD
~~~
python train.py --data_dir PATH_TO_DATASET \
--data CIFAR100 --batch_size 256 --alpha 3 --beta 100 \
~~~