import streamlit as st
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
import torchvision.transforms as T
from Filters import plot_filters_multi_channel, plot_filters_single_channel_big, plot_filters_single_channel, plot_weights
from GradCam import process_image, visualize, Feat_Extractor, Netout, GradientCam
from Occlusion import occlusion
from SimpleGradient import SimpleGradient
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import streamlit as st 

#defining the transformations for the data

from PIL import Image 

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

intro_markdown = read_markdown_file("main.md")
st.markdown(intro_markdown, unsafe_allow_html=True)
def main():
    st.sidebar.title("Choose one of the Methods")
    app_mode = st.sidebar.selectbox("",
        ["Choose one of the Methods from Below", "Occulusion Method", "Filter Visualization", "Simple Gradient Method", "Grad Cam Method"])

    if app_mode == "Occulusion Method":
        intro_markdown = read_markdown_file("Occlusion/occlusion.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
        Occulusion_Method()

    elif app_mode == "Filter Visualization":
        intro_markdown = read_markdown_file("Filters/filters.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
        cnn_filters()


    elif app_mode == "Simple Gradient Method":
        intro_markdown = read_markdown_file("SimpleGradient/SimpleGradient.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
        simple_grad()
        
    elif app_mode == "Grad Cam Method":
        st.text("This picture shows the basic working pricpipel behind gradcam.")
        st.image("GradCam/gradcam.jpeg", width = 900)
        intro_markdown = read_markdown_file("GradCam/gradcam.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
        grad_cam()

def Occulusion_Method():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    ref_image = st.file_uploader("Reference Image", ['jpeg', 'png', 'jpg'], None)

    if ref_image:
        st.image(ref_image, caption="Reference image")

    process = st.button("Process")
    if process:
        i = Image.open(ref_image)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #normalize the images with imagenet data mean and std
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        ii = transform(i)
        iii = ii.reshape(1,3, 224, 224)
        model = models.vgg16(pretrained=True)
        model.eval()

        #running inference on the images without occlusion
        #vgg16 pretrained model
        outputs = model(iii)

        #passing the outputs through softmax to interpret them as probability
        outputs = nn.functional.softmax(outputs, dim = 1)

        #getting the maximum predicted label
        prob_no_occ, pred = torch.max(outputs.data, 1)

        #get the first item
        prob_no_occ = prob_no_occ[0].item()
        heatmap = occlusion(model, iii, pred[0].item(), 32, 14)
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatmap, xticklabels =False, yticklabels = False, vmax= prob_no_occ )
        st.pyplot(fig)
        # st.image(imgplot)

def cnn_filters():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    ref_image = st.file_uploader("Reference Image", ['jpeg', 'png', 'jpg'], None)
    if ref_image:
        st.image(ref_image, caption="Reference image")
    alexnet = models.alexnet(pretrained = True)
    filter_number = (st.number_input("Enter the Conv Layer Number"))
    single_channel = st.text_input("Is the filter Single channel ? True or False")
    filter_number = int(filter_number)
    process = st.button("Process")

    if process:
        plot_weights(alexnet, filter_number, single_channel = single_channel)

def grad_cam():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    ref_image = st.file_uploader("Reference Image", ['jpeg', 'png', 'jpg'], None)
    if ref_image:
        st.image(ref_image, caption="Reference image")
    # Opening the image
    process = st.button("Process")

    if process: 
        img = Image.open(ref_image)
        image = Image.open(ref_image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image = np.float32(cv2.resize(image, (224, 224))) / 255
        input = process_image(image)

        net = models.resnet50(pretrained=True)
        grad_cam = GradientCam(model=net, feature_module=net.layer4, \
                            target_layer_=["2"], cuda=False)

        target_index = None

        weights, class_activation_map = grad_cam(input, target_index)

        visualize(image, class_activation_map)

def simple_grad():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    ref_image = st.file_uploader("Reference Image", ['jpeg', 'png', 'jpg'], None)
    if ref_image:
        st.image(ref_image, caption="Reference image")
    process = st.button("Process")

    if process:
        img = Image.open(ref_image)
        SimpleGradient(img)

    
if __name__ == "__main__":
    main()