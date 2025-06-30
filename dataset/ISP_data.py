import numpy as np
from pathlib import Path
import torch.utils.data as data
import torch
from torchvision import transforms
import cv2


class ISPData(data.Dataset):
    def __init__(
        self,
        data_dir,
        stage="train",
    ):
        # Set all input args as attributes
        # ...
        self.stage = stage

        if stage != "test":
            # training or validation stage
            pass
        else:
            pass

    def __len__(self):
        if self.stage == "test":
            return #len()
        else:
            return #len()

    def __getitem__(self, idx):
        if self.stage != "test":
            return # train data batch 
        else:
            return # test data batch
