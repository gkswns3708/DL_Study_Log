import os
import cv2

from glob import glob 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class CategroyDataset(Dataset):
    def __init__(self, text, image_path, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)

    def __getitem(self, index):
        text = str(self.text[index])
        image_path = os.path.join('../', str(self.image_path[index][2:]))
        image = cv2.imread(image_path)
        encoding = 
