import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
import torch 
import PIL 
import os 
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

import streamlit as st 

# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def SimpleGradient(img):
    model = torchvision.models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False


    # preprocess the image
    X = preprocess(img)
    model.eval()
    X.requires_grad_()
    logits = model(X)

    # Get the index corresponding to the maximum score and the maximum score itself.
    logits_max_index = logits.argmax()
    logits_max = logits[0,logits_max_index]
    logits_max.backward()

    saliency_map, _ = torch.max(X.grad.data.abs(),dim=1)
    print (torch.mean(saliency_map),torch.max(saliency_map))
    # code to plot the saliency map as a heatmap
    fig = plt.figure(figsize=(20,20))

    plt.imsave('tt.png', saliency_map[0], cmap=plt.cm.hot)
    st.image('tt.png')
    os.remove('tt.png')

  