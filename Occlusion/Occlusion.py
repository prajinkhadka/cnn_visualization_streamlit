import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import torch 
import PIL 
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import streamlit as st 
from PIL import Image 

def occlusion(model, image, label, occ_size =50 , occ_stride=50, occ_pixel = 0.5):
    width, height = image.shape[-2], image.shape[-1]

    output_height = int(np.ceil((height -occ_size)/occ_stride))
    output_width = int(np.ceil((width - occ_size)/ occ_stride))

    heatmap = torch.zeros((output_height, output_width))

    for h in range(0, height):
        for w in range(0, width):
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            if (w_end) >= width or (h_end) >= height:
                continue
            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]
            heatmap[h, w] = prob 
    return heatmap
