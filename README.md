# Sample code for the Class Activation Mapping
We propose a simple technique to expose the implicit attention of Convolutional Neural Networks on the image. It highlights the most informative image regions relevant to the predicted class. You could get attention-based model instantly by tweaking your own CNN a little bit more. The paper is published at [CVPR'16](http://arxiv.org/pdf/1512.04150.pdf).

The framework of the Class Activation Mapping is as below:
![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)

Some predicted class activation maps are:
![Results](http://cnnlocalization.csail.mit.edu/example.jpg)

### NEW: PyTorch Demo code
* The popular networks such as ResNet, DenseNet, SqueezeNet, Inception already have global average pooling at the end, so you could generate the heatmap directly without even modifying the network architecture. Here is a [sample script](pytorch_CAM.py) to generate CAM for the pretrained networks.
```
    python pytorch_CAM.py
```
You also could take a look at the [unified PlacesCNN scene prediction code](https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py) to see how the CAM along with scene categories, scene attributes are predicted. It has been used in the [PlacesCNN scene recognition demo](http://places2.csail.mit.edu/demo.html).

### Pre-trained models in Caffe:
* GoogLeNet-CAM model on ImageNet: ```models/deploy_googlenetCAM.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/imagenet_googlenetCAM_train_iter_120000.caffemodel]
* VGG16-CAM model on ImageNet: ```models/deploy_vgg16CAM.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/vgg16CAM_train_iter_90000.caffemodel]
* GoogLeNet-CAM model on Places205: ```models/deploy_googlenetCAM_places205.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/places_googlenetCAM_train_iter_120000.caffemodel]
* AlexNet+-CAM on ImageNet:```models/deploy_alexnetplusCAM_imagenet.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/alexnetplusCAM_imagenet.caffemodel]
* AlexNet+-CAM on Places205 (used in the [online demo](http://places.csail.mit.edu/demo.html)):```models/deploy_alexnetplusCAM_places205.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/alexnetplusCAM_places205.caffemodel]

### Usage Instructions:
* Install [caffe](https://github.com/BVLC/caffe), compile the matcaffe (matlab wrapper for caffe), and make sure you could run the prediction example code classification.m.
* Clone the code from Github:
```
git clone https://github.com/metalbubble/CAM.git
cd CAM
```
* Download the pretrained network
```
sh models/download.sh
```
* Run the demo code to generate the heatmap: in matlab terminal, 
```
demo
```
* Run the demo code to generate bounding boxes from the heatmap: in matlab terminal,
```
generate_bbox
```

The demo video of what the CNN is looking is [here](https://www.youtube.com/watch?v=fZvOy0VXWAI). The reimplementation in tensorflow is [here](https://github.com/jazzsaxmafia/Weakly_detector). The pycaffe wrapper of CAM is reimplemented at [here](https://github.com/gcucurull/CAM-Python).

### ILSVRC evaluation

* See the script [ILSVRC_evaluate_bbox.m](ILSVRC_evaluate_bbox.m) and [ILSVRC_generate_heatmap.m](ILSVRC_generate_heatmap.m) if you want to reproduce the ILSVRC localizatgion result. Also see [this file](http://cnnlocalization.csail.mit.edu/ILSVRC_evaluation_raw.tar.gz) for some intermedate files. You still need to download the ILSVRC toolkit at http://image-net.org/challenges/LSVRC/2012/. The code is written in a rush and without any clean-up, Please figure out how to set up things
  properly by yourself.

### Reference:
```
@inproceedings{zhou2016cvpr,
    author    = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
    title     = {Learning Deep Features for Discriminative Localization},
    booktitle = {Computer Vision and Pattern Recognition},
    year      = {2016}
}
```
### License:
The pre-trained models and the CAM technique are released for unrestricted use.

Contact [Bolei Zhou](http://people.csail.mit.edu/bzhou/) if you have questions.
    
