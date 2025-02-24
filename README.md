# Offline Safe Reinforcement Learning Using Trajectory Classification
in Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25).

[**[Project Page]**](https://trac.github.io) [**[Arxiv]**](https://arxiv.org/abs/2412.15429)

The official implementation of TraC, which trains a policy to generate desirable trajectories and avoid undesirable ones while bypassing min-max optimization and traditional RL. We employ a two-phase approach:

1. Creating desirable and undesirable datasets;
2. Direct policy learning via trajectory classification.


## Installation
``` Bash
conda env create -f environment.yml
conda activate TraC
git clone https://github.com/zgong11/TraC.git
cd TraC
```

## Run experiments

To train a model, simply run the scripts in `scripts/run_[env].sh` after activating the environment. For example:

### BulletSafetyGym
```
./scripts/run_bulletgym.sh BallRun 1.0 0
```

### SafetyGym
```
./scripts/run_safetygym.sh PointButton1 0.75 0
```

### MetaDrive
```
./scripts/run_metadrive.sh easydense 0.25 0
```


## Bibtex

If you find our code or paper can help, feel free to cite our paper as:
```
@inproceedings{
gong2025offline,
title={Offline Safe Reinforcement Learning Using Trajectory Classification},
author={Gong, Ze and Kumar, Akshat and Varakantham, Pradeep},
booktitle={The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI)},
year={2025}
}
```


## Acknowledgements

Parts of this code are adapted from [CPL](https://github.com/jhejna/cpl).