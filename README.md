# One-Shot Classification with Omniglot Dataset

This repository includes a TensorFlow implementation of a convolutional neural network (CNN) for one-shot classification of images in the [Omniglot](https://github.com/brendenlake/omniglot) dataset. I am attempting to replicate the performance in [Lake et. al., ''Human-level concept learning through probabilistic program induction''](http://www.sciencemag.org/content/350/6266/1332.short), which achieved 13.5% error with a CNN classifier.

Thus far I have gotten close to 30% error, but still have a ways to go to match the performance in Lake et. al.

The CNN is first trained on images from the 30 background alphabets of the Omniglot dataset. The hidden layer preceding the final softmax layer is then used to define a classifier for images from the 20 evaluation alphabets.  Given two input images, the [dot product](https://en.wikipedia.org/wiki/Cosine_similarity) of their feature representations in the hidden layer quantifies their similarity. The classifier uses this distance metric to match images of the same character.

Optimization: 
I used the Adam optimizer with a learning rate of 0.001, a dropout rate of 0.4 (for the dense layer), L2 regularization with coefficient 0.0005. (I used different learning rates for a small network and training set, and found 0.001 -- within a factor of 3 -- to minimize the validation cost at early stopping.  I did not find significant variation of the final validation cost with the regularization parameters, so I used the values from Lake et. al.)

Architecture. 
I explored architectures of the same shape as the CNN found in Lake et. al. to give the best performance out of seven architectures, but with varying overall size.  Specifically, I used a scale parameter s, which fixes the number of channels in the convolutional layers.  The image data are fed into a pooling layer to downsample the images to a lower resolution, which is followed by two convolution layers with 12s and 30s channels, respectively.  The output is fed into a second pooling layer with the pooling size and stride of 2.  This pooling layer is followed by a dense layer with 300s channels, which is followed by a softmax layer with N units, depending on the training set size (see below). Note that s=10 recovers the architecture of Lake et. al.

To generate classifiers with differing degrees of expressive power, I pre-trained CNNs on training sets of different size, and adjusted the network size accordingly.  Specifically, I trained on the first N={50,100,200,500,964} characters in the 964-character omniglot dataset of background images, and for each N I trained CNNs with different values of the scale parameter s, finding the network size s(N) at which learning saturates due to finite amount of data available.  I used early stopping to terminate training when the validation cost approached an asymptotic value.

A few lines of code are taken from [github.com/brendenlake/omniglot/blob/master/python/one-shot-classification/demo_classification.py](https://github.com/brendenlake/omniglot/blob/master/python/one-shot-classification/demo_classification.py).
