# GapNet-PL
This repository contains code to reproduce the results of "Human-level Protein Localization with Convolutional Neural Networks".

```
@inproceedings{
rumetshofer2018humanlevel,
title={Human-level Protein Localization with Convolutional Neural Networks},
author={Elisabeth Rumetshofer and Markus Hofmarcher and Clemens Röhrl and Sepp Hochreiter and Günter Klambauer},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryl5khRcKm},
}
```

# Dataset
The dataset with individual samples in NPZ-format is available for download here: https://ml.jku.at/software/pLoc/dataset
A similar dataset was analyzed during a Kaggle challenge: https://www.kaggle.com/c/human-protein-atlas-image-classification
One of the gold medal winners (Wienerschnitzelgemeinschaft) used an architecture based on GapNet as described here:
https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77300

# Instructions

Setup an environment with the following dependencies:
```
conda install -c tensorflow -c pytorch tensorflow-gpu=1.8.0 pytorch=0.4.0 cudatoolkit=9.0 tqdm matplotlib scikit-learn natsort numpy torchvision 
``` 
This code uses a small library to simplify experiment archiving we call pyll.
Therefore, before running the code either unzip the library from pyll-0.1.tar.gz into a folder of your choice and set the 
PYTHONPATH accordingly or simply install it via pip with
```
pip install pyll-0.1.tar.gz
```
## Configs
We use configuration files to set hyperparameters and directories, sample configurations are provided in the configs folder
and have to be adjusted accordingly.
Parameters from the configuration files can also be overwritten from the command line.

## Training
```
python main.py --config <configfile> --cuda_gpu <gpu-id> --num_workers <number of dataloading threads> --batchsize <bs>
```
