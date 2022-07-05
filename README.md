# SemanticRL

![](./teaser.png)

**SemanticRL is designed to preserve the semantic information instead of strictly securing the bit-level precision. It enables a general-purpose, large-scale, wireless, and semantic communication framework.**

## Features
+ A schematic shift from bit-precision to semantic consistency.
+ Compatible with any (non-differenable) semantic similarity metric as the objective function.
+ RL-based end-to-end optimization on non-differentiable and unknown wireless channel with high-dimensional action/semantic space.


## Requirements
```
pip install -r requirements.txt
```


## Dataset Preparation
```
cd $your_data_root
wget https://www.statmt.org/europarl/v7/fr-en.tgz
tar -zxvf fr-en.tgz    
python preprocess_captions.py --data_root $your_data_root
```


## Training

### Training SemanticRL-JSCC

```
AWGN-CE-Stage1
python Trainng_SemanticRL.py --training_config ./config/config_AWGN_CE.yaml --dataset_path $your_data_root

AWGN-CE-Stage2
python Trainng_SemanticRL.py --training_config ./config/config_AWGN_CE_Stage2.yaml --dataset_path $your_data_root

AWGN-RL-Stage2
python Trainng_SemanticRL.py --training_config ./config/config_AWGN_RL.yaml --dataset_path $your_data_root
```

You can change the type of random channel to trian and test in different scenarios. For more details, run `python Trainng_SemanticRL.py --help`.


### Training SemanticRL-SCSIU
```
python Trainng_SemanticRL.py --training_config ./config/config_AWGN_RL_SCSIU.yaml --dataset_path $your_data_root
```


## Integrating SemanticRL with your own framework

Besides `LSTM` backbone, we provide a `Transformer` backbone to facilitate further researches. You can rewrite methods in `model.py` to customize your own framework. SemanticRL is model-agnostic. You may also design any semantic similarity metric to build a customed communication system.


## Timeline

First submission, to JSAC-Series on Machine Learning for Communications and Networks (2021.10.26)
https://arxiv.org/abs/2108.12121v1
Rating 5,3,5 (higher the better)
-Rejected


Revised, submitted to JSAC-Special Issue on Beyond Transmitting Bits (2022.03.31)
https://arxiv.org/abs/2108.12121v2
Rating 7,3,8,6 (higher the better)
-Rejected

I feel deeply appreciated that you arrived here. Inspite of few negative comments, the authors are greatly encouraged by the dedicated reviewers, along with their constructive suggestions which keep improving this work. 

## Thanks

This repository is largely inspired by [ruotianluo's excellent captioning work](https://github.com/ruotianluo/ImageCaptioning.pytorch).


## Citation
```
@article{lu2021reinforcement,
  title={Reinforcement learning-powered semantic communication via semantic similarity},
  author={Lu, Kun and Li, Rongpeng and Chen, Xianfu and Zhao, Zhifeng and Zhang, Honggang},
  journal={arXiv preprint arXiv:2108.12121},
  url={https://arxiv.org/abs/2108.12121},
  year={2021}
}
```




