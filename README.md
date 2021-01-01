# CNN Visualization Streamlit

I have created this webapp to Visualize the trained CNN network( VGG19, Resnet, Alexnet ) on Imagenet dataset i.e how exactly CNN predicts class label, what parts of the images are responsible for certain class.
You can upload your own picture and see the result.

For now, these are following methods
1. Occulsion method
2. Visualzing each conv layer and filters
3. Simple Gradient method
4. GradCam method.

With these you can visualize which parts of the image ws responsible to preciting the class label.

link - https://lnkd.in/gMPNdS9


PS - Right now, I have deployed in a small instance so site may be little slow ( will try to scale up later )


For building the project locally, 

1. Install [Docker](https://www.docker.com/)
2. Clone the repo 
   ```git clone https://github.com/prajinkhadka/cnn_visualization_streamlit.git ```
4. Navigate to the Project Directory.
   ``` cd home/prajin/ProjectDirectory ```
3. Build the Docker Image 
  ``` docker build -t AppName:tag .  ```
4. Run the docker image in the container 
   ``` docker run -p 8501:8501 AppName:tag ```


Please help in improving the design or adding more methods for visualziation or improving the current one or adding more pretrained models.
