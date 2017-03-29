# KERAS-DCGAN

Example implementation of adversarial agents with keras. 

## Example 1 dcgan 

For generating artificial images with deep learning.

This trains two adversarial deep learning models on real images, in order to
produce artificial images that look real.

## Agents:

- A **generator** which takes a random signal + a digit class
- A **discriminator** which is given images and has to decide if they are fake or not
- A **classifier** which is given generated images and classifies them with a digit class


The generator model tries to produce images that look real and get a high score from the discriminator.



The discriminator model tries to tell apart between real images and artificial images from the generator.



## Usage


**Training:**

```
python dcgan.py --mode train --batch-size <batch_size>

$ python dcgan.py --mode train --path ~/generated-images --batch_size 256
```


**Image generation:**

```
python dcgan.py --mode generate --batch_size <batch_size>

$ python dcgan.py --mode generate --batch_size 64 --nice` : top 5% images according to discriminator

$ python dcgan.py --mode generate --batch_size 128
```


## Result


**generated images :** 


![generated_image.png](./assets/generated_image.png) ![nice_generated_image.png](./assets/nice_generated_image.png)


**train process :**


![training_process.gif](./assets/training_process.gif)


## Acknowledgments

Based on [jacobgil](https://github.com/jacobgil/keras-dcgan)'s implementation of 
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).

High level changes from that repository:

- Adds a third agent which classifies generated digits
- Generator given an input class as well as random IV


