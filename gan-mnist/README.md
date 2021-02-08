# GAN MNIST

GAN project on MNIST dataset from Udacity Deep Learning nanodegree exercise. The notebook shows how to generate new handwritten digits. 

## About the project

The project shows how to build a generative adversarial network (GAN) trained on MNIST dataset. 

GANs were [first reported on](https://arxiv.org/abs/1406.2661) in 2014 from Ian Goodfellow and others in Yoshua Bengio's lab. Since then, GANs have exploded in popularity.

The idea behind GANs is that you have two networks, a generator $G$ and a discriminator $D$, competing against each other. The generator makes "fake" data to pass to the discriminator. The discriminator also sees real training data and predicts if the data it's received is real or fake.

* The generator is trained to fool the discriminator, it wants to output data that looks as close as possible to real, training data.
* The discriminator is a classifier that is trained to figure out which data is real and which is fake.

What ends up happening is that the generator learns to make data that is indistinguishable from real data to the discriminator.

<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/gan_pipeline.png">

The general structure of a GAN is shown in the diagram above, using MNIST images as data. The latent sample is a random vector that the generator uses to construct its fake images. This is often called a latent vector and that vector space is called latent space. 
As the generator trains, it figures out how to map latent vectors to recognizable images that can fool the discriminator.

## The Model Definition

A GAN is comprised of two adversarial networks, a discriminator and a generator.

### Discriminator

The discriminator network is going to be a pretty typical linear classifier. To make this network a universal function approximator, we'll need at least one hidden layer, and these hidden layers should have one key attribute:

> All hidden layers will have a [Leaky ReLu](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU) activation function applied to their outputs.

<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/gan_network.png">

#### Leaky ReLU

We should use a leaky ReLU to allow gradients to flow backwards through the layer unimpeded. A leaky ReLU is like a normal ReLU, except that there is a small non-zero output for negative input values.

<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/leaky_relu.png" width=40%>

#### Sigmoid Output
We'll also take the approach of using a more numerically stable loss function on the outputs. Recall that we want the discriminator to output a value 0-1 indicating whether an image is real or fake.

>  We will ultimately use [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss), which combines a `sigmoid` activation function **and** and binary cross entropy loss in one function. 


### Generator

The generator network will be almost exactly the same as the discriminator network, except that we're applying a [tanh activation function](https://pytorch.org/docs/stable/nn.html#tanh) to our output layer.

#### tanh Output 

The generator has been found to perform the best with $tanh$ for the generator output, which scales the output to be between -1 and 1, instead of 0 and 1. 

<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/tanh_fn.png" width=40%>

## Training results

```
Epoch [  100/  100] | d_loss: 1.3607 | g_loss: 1.2538
```
<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/loss.PNG">

Below I'm showing the generated images as the network was training, every 10 epochs.

<img src="https://github.com/Shahrullo/GAN/blob/main/gan-mnist/assets/sample.PNG">

It starts out as all noise. Then it learns to make only the center white and the rest black. You can start to see some number like structures appear out of the noise like 1s and 9s.







