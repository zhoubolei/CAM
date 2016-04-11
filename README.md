# Sample code for the Class Activation Mapping
We propose a simple technique to espose the implicit attention of Convolutional Neural Networks on the image. It highlights the most informative image regions relevant to the predicted class. You could get attention-based model instantly by tweaking your own CNN a little bit more.

![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)

### Usage Instructions:
1. Install [caffe](https://github.com/BVLC/caffe), compile the matcaffe (matlab wrapper for caffe), and make sure you could run the prediction example code classification.m.
2. In matlab, run demo.m.

The demo video of what the CNN is looking is [here](https://www.youtube.com/watch?v=fZvOy0VXWAI). The reimplementation in tensorflow is [here](https://github.com/jazzsaxmafia/Weakly_detector).

### Reference:
    B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba
    Learning Deep Features for Discriminative Localization.
    Computer Vision and Pattern Recognition (CVPR), 2016
    [PDF](http://arxiv.org/pdf/1512.04150.pdf)[Project Page](http://cnnlocalization.csail.mit.edu/)

Contact [Bolei Zhou](http://people.csail.mit.edu/bzhou/) if you have questions.
    
