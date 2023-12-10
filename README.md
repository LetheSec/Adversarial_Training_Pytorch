# Adversarial Training Library - Pytorch

Pytorch implementation of commonly used adversarial training methods, including:

- PGD-AT [1]
- TRADES [2]
- LBGAT [3]
- MART [4]
- STAT [5]
- AT-AWP [6] 
- TRADES-AWP [6]
- STAT-AWP [5]

Checkpoints can be downloaded here: [百度网盘](https://pan.baidu.com/s/16Zt5fmHgPWAJrb9WGGTNIQ?pwd=w5rr) []()

## Training from Scratch

You can train from scratch as follows

```bash
python normal_train.py --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python at.py           --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python trades.py       --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python lbgat.py        --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python mart.py         --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python stat.py         --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python at_awp.py       --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python trades_awp.py   --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
python stat_awp.py     --data [CIFAR10|CIFAR100] --arch [ResNet18|WideResNet]
```

## Evaluate 

You can perform a robustness evaluation on the pretrained weights:

```bash
python evaluate.py --data CIFAR10 --arch ResNet18 \
--checkpoint ./adversarial_training_checkpoints/stat_awp_resnet18_cifar10.tar
```

## Results

All results are selected with the best robustness on the test set.


### ResNet18 - CIFAR10


|     Method     | Clean Acc | FGSM  | PGD-20 | PGD-100 | CW-100 |
| :------------: | :-------: | :---: | :----: | :-----: | :----: |
|    Natural     |   95.38   | 95.38 |  0.02  |  0.00   |  0.00  |
|   PGD-AT [1]   |   82.57   | 56.99 | 51.28  |  51.02  | 49.66  |
|   TRADES [2]   |   80.86   | 55.91 | 51.23  |  51.14  | 48.18  |
|   LBGAT [3]    |   76.40   | 56.69 | 53.19  |  52.98  | 50.41  |
|    MART [4]    |   78.59   | 58.54 | 54.87  |  54.58  | 49.14  |
|    STAT [5]    |   83.39   | 59.52 | 53.85  |  53.58  | 50.96  |
|   AT-AWP [6]   |   82.30   | 59.68 | 54.73  |  54.39  | 52.05  |
| TRADES-AWP [6] |   81.90   | 58.96 | 55.76  |  55.68  | 52.36  |
|  STAT-AWP [5]  |   83.03   | 60.20 | 56.35  |  56.25  | 52.64  |

### ResNet18 - CIFAR100

|     Method     | Clean Acc | FGSM  | PGD-20 | PGD-100 | CW-100 |
| :------------: | :-------: | :---: | :----: | :-----: | :----: |
|    Natural     |   78.70   | 8.30  |  0.00  |  0.00   |  0.00  |
|   PGD-AT [1]   |   56.76   | 31.96 | 28.91  |  28.89  | 26.89  |
|   TRADES [2]   |   54.77   | 30.02 | 27.90  |  27.70  | 24.33  |
|   LBGAT [3]    |   57.07   | 34.70 | 32.50  |  32.43  | 27.32  |
|    MART [4]    |   54.20   | 34.48 | 32.19  |  32.15  | 27.99  |
|    STAT [5]    |   57.86   | 32.97 | 30.55  |  30.46  | 26.47  |
|   AT-AWP [6]   |   58.75   | 35.47 | 32.57  |  32.50  | 30.00  |
| TRADES-AWP [6] |   59.06   | 34.13 | 31.77  |  31.76  | 27.62  |
|  STAT-AWP [5]  |   58.35   | 34.83 | 32.69  |  32.63  | 27.95  |

### WideResNet-34-10 - CIFAR100

|     Method     | Clean Acc | FGSM  | PGD-20 | PGD-100 | CW-100 |
| :------------: | :-------: | :---: | :----: | :-----: | :----: |
|    Natural     |   96.31   | 55.30 |  0.04  |  0.00   |  0.00  |
|   PGD-AT [1]   |   86.48   | 61.16 | 54.89  |  54.42  | 54.10  |
|   TRADES [2]   |   84.38   | 60.50 | 54.58  |  54.40  | 53.13  |
|   LBGAT [3]    |   79.47   | 60.56 | 56.74  |  56.58  | 54.08  |
|    MART [4]    |   83.01   | 61.65 | 56.72  |  56.36  | 53.15  |
|    STAT [5]    |   84.75   | 62.23 | 57.29  |  57.10  | 54.93  |
|   AT-AWP [6]   |   85.60   | 63.18 | 57.84  |  57.54  | 55.92  |
| TRADES-AWP [6] |   84.90   | 62.87 | 59.19  |  59.09  | 55.99  |
|  STAT-AWP [5]  |   86.41   | 64.83 | 60.31  |  60.03  | 56.58  |

## Reference

[1] [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) 

[2] [Theoretically Principled Trade-off between Robustness and Accuracy](http://arxiv.org/abs/1901.08573)

[3] [Learnable Boundary Guided Adversarial Training](https://arxiv.org/abs/2011.11164v2)

[4] [Improving Adversarial Robustness Requires Revisiting Misclassified Examples](https://openreview.net/forum?id=rklOg6EFwS)

[5] [Squeeze Training for Adversarial Robustness](https://arxiv.org/abs/2205.11156)

[6] [Adversarial Weight Perturbation Helps Robust Generalization](http://arxiv.org/abs/2004.05884)
