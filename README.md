# Sample code for the Class Activation Mapping
We propose a simple technique to espose the implicit attention of Convolutional Neural Networks on the image. It highlights the most informative image regions relevant to the predicted class. You could get attention-based model instantly by tweaking your own CNN a little bit more. The paper is published at [CVPR'16](http://arxiv.org/pdf/1512.04150.pdf).

The framework of the Class Activation Mapping is as below:
![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)

Some predicted class activation maps:
![Results](http://cnnlocalization.csail.mit.edu/example.jpg)

### Pre-trained models:
* GoogLeNet-CAM model on ImageNet: ```models/deploy_googlenetCAM.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel]
* VGG16-CAM model on ImageNet: ```models/deploy_vgg16CAM.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/vgg16CAM_train_iter_90000.caffemodel]
* GoogLeNet-CAM model on Places205: ```models/deploy_googlenetCAM_places205.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/places_googleletCAM_train_iter_120000.caffemodel]
* AlexNet-CAM on Places205 (used in the [online demo](http://places.csail.mit.edu/demo.html)):```models/deploy_alexnetplusCAM_places205.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/alexnetplusCAM_places205.caffemodel]

### Usage Instructions:
1. Install [caffe](https://github.com/BVLC/caffe), compile the matcaffe (matlab wrapper for caffe), and make sure you could run the prediction example code classification.m.
2. In matlab, run demo.m.

The demo video of what the CNN is looking is [here](https://www.youtube.com/watch?v=fZvOy0VXWAI). The reimplementation in tensorflow is [here](https://github.com/jazzsaxmafia/Weakly_detector).

### Reference:
    B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba
    Learning Deep Features for Discriminative Localization.
    Computer Vision and Pattern Recognition (CVPR), 2016

### License:
The pre-trained models and techniques could be used without constraints.

Contact [Bolei Zhou](http://people.csail.mit.edu/bzhou/) if you have questions.
    
