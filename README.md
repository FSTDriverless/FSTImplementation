# Formula Student Technion Driverless - Implementation  

This repository introduces the procedure of implementing a model, trained by [FSTDriverless/AirSim](https://github.com/FSTDriverless/AirSim) using Keras, on Nvidia Jetson TX2.  

This procedure is composed of a few steps:  
1. Freeze a trained Keras model (TensorFlow backend).  
2. Convert the frozen model to a TensorFlow model.  
3. Load the TensorFlow model and perform inference using an IDS camera with the TX2.  

Our purpose was to send the model's output to an Arduino module. In case you don't need this option, you can remove any dependency on serial port communication.  

![drone](drone.gif)


## Prerequisites  

If you already has a frozen TensorFlow model, you need to use only the Jetson's prerequisites.  

### x86 PC contains:  
* Operating system: Windows 10  
* GPU: Nvidia GTX 1080 or higher (recommended)  
* Development: CUDA 9.0 and python 3.5.  
* Python libraries: Keras 2.1.2, TensorFlow 1.6.0.  
* Note: Newer versions of keras or tensorflow are recommended but can cause syntax errors.  

### Jetson TX2 contains:  
* Operating system: Ubuntu 16.04  
* Drivers: [IDS camera driver](https://en.ids-imaging.com/download-ueye-emb-softfloat.html).  
* Development: CUDA 9.0 and python 3.5.  
* Python libraries: [TensorFlow 1.8.0](https://devtalk.nvidia.com/default/topic/1031300/tensorflow-1-8-wheel-with-jetpack-3-2-/), [OpenCV 3.4.1](https://www.jetsonhacks.com/2018/05/28/build-opencv-3-4-with-cuda-on-nvidia-jetson-tx2/), numpy, pyueye, pyserial.  


## What's inside  
### What's new:   
* We now share our TF frozen trained model. The model can be found in [Models folder](Models).   

This repository contains two code files:  

**freezing_keras_to_tf.py**  
The main actions in this script are uploading a Keras model, freezing it and saving it as a TensorFlow model.  
The code rely on having an adjacent folder named "models" contains a keras model named "model.h5". The output model will be stored in this folder as well.  

Note: you'll have to understand what is your output node. In our case, it was the last activation function in our model, so our output node was "output/Sigmoid".  
To print the list of your model nodes in Keras, add the following command:  
```
[print(n.name) for n in tf.get_default_graph().as_graph_def().node]  
```

**inference.py**  
Using a frozen Tensorflow model, this script gets images from the IDS camera, predict the corresponding steering angle using the model and sending it to the Arduino module.  
The code rely on having a model named "model_tf.pb" in the same folder.  
The output in our case is a prediction in a range of [0,255].  

The code implements an inference for PilotNet architecture. Therefore, we adjusted the image to adapt the network. It's recommended to read the code carefully.  


### Formula Student Technion team  

[Tom Hirshberg](https://www.linkedin.com/in/tom-hirshberg-93935b16b/) and [Dean Zadok](https://www.linkedin.com/in/dean-zadok-36886791/).  

### Acknowledgments  

We would like to thank our advisors: Dr. Kira Radinsky, Dr. Ashish Kapoor, Boaz Sternfeld and David Dovrat.  
Thanks to the Intelligent Systems Lab (ISL) in the Technion for their support.  
