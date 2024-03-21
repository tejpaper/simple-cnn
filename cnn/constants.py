import os
import torch

RANDOM_SEED = 1337
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOGS_DIR = 'logs'
DATA_DIR = 'data'
""" 
Data source: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
Dataset structure is as follows:
    .
    ├── interface.tar
    ├── Parasitized
    │   └── ...
    └── Uninfected
        └── ...
"""
DATA_INTERFACE_PATH = os.path.join(DATA_DIR, 'interface.tar')
