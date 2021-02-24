# Deep Convolution Generative Adversarial Networks - [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)

DCGAN is a GAN using convolutional layers in the generator and discriminator. The DCGAN architecture was first explored in 2016 and has seen impressive results in generating new images.

In this notebook, we tried to train DCGAN on the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) (SVHN) dataset. These are color images of house numbers collected from Google street view. SVHN images are in color and much more variable than MNIST.

![SVHN](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/svhn_dcgan.png)


## Defining The Model
A GAN is comprised of two adversarial networks, a discriminator and a generator.

### Discriminator

The Discriminator is a convolutional classifier without any maxpooling layers

Here is the architecture of our model

![Discriminator](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/conv_discriminator.png)

We also use batch normalization [nn.BatchNorm2d](https://pytorch.org/docs/stable/nn.html#batchnorm2d) on each layer **except** the first convolutional layer and final, linear output layer.  

### Generator
