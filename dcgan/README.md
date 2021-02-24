# Deep Convolution Generative Adversarial Networks - [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)

DCGAN is a GAN using convolutional layers in the generator and discriminator. The DCGAN architecture was first explored in 2016 and has seen impressive results in generating new images.

In this notebook, we tried to train DCGAN on the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) (SVHN) dataset. These are color images of house numbers collected from Google street view. SVHN images are in color and much more variable than MNIST.

![SVHN](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/svhn_dcgan.png)


## Defining The Model
A GAN is comprised of two adversarial networks, a discriminator and a generator.

### Discriminator

The Discriminator is a convolutional classifier without any maxpooling layers

* The inputs to the discriminator are 32x32x3 tensor images
* A few convolutional, hidden layers
* Last a fully connected layer for the output; we want a [sigmoid](https://pytorch.org/docs/stable/nn.functional.html?highlight=sigmoid#torch.nn.functional.sigmoid) output, but we'll add that in the loss function, [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)

Here is the architecture of our discriminator

![Discriminator](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/conv_discriminator.png)

We also use batch normalization [nn.BatchNorm2d](https://pytorch.org/docs/stable/nn.html#batchnorm2d) on each layer **except** the first convolutional layer and final, linear output layer.  

### Generator

In Generator the input will be our noise vector *z. And the output will be a $tanh$ output with size 32x32 which is the size of SVHN images.

* The first layer is a fully connected layer which is reshaped into a deep and narrow layer, something like 4x4x512.
* Batch normalization and a leaky ReLU activation.
* Next is a series of [transpose convolutional layers](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d), where typically halve the depth and double the width and height of the previous layer.
* And, apply batch normalization and ReLU to all but the last of these hidden layers. Where it will be just applied a tanh activation.

![Generator](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/conv_generator.png)


## Train
Training will involve alternating between training the discriminator and the generator.

### Discriminator training
1. Compute the discriminator loss on real, training images
2. Generate fake images
3. Compute the discriminator loss on fake, generated images
4. Add up real and fake loss
5. Perform backpropagation + an optimization step to update the discriminator's weights

### Generator training
1. Generate fake images
2. Compute the discriminator loss on fake images, using **flipped** labels!
3. Perform backpropagation + an optimization step to update the generator's weights

### Training loss

Plot of the training losses for the generator and discriminator, recorded after each epoch.
![train_loss](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/TrainlLoss.PNG)


### Generator samples from training
Here we can view samples of images from the generator. We'll look at the images we saved during training.

![generated images](https://github.com/Shahrullo/GAN/blob/main/dcgan/assets/generatedexamples.PNG)
