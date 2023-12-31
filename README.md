# GACNN: Training Deep Convolutional Neural Networks with Genetic Algorithm

This repository contains the code and data for the paper "GACNN: Training Deep Convolutional Neural Networks with Genetic Algorithm" by Parsa Esfahanian and Mohammad Akhavan.

## Abstract

Convolutional Neural Networks (CNNs) have gained a significant attraction in the recent years due to their increasing real-world applications. Their performance is highly dependent to the network structure and the selected optimization method for tuning the network parameters. In this paper, we propose novel yet efficient methods for training convolutional neural networks. The most of current state of the art learning method for CNNs are based on Gradient decent. In contrary to the traditional CNN training methods, we propose to optimize the CNNs using methods based on Genetic Algorithms (GAs). These methods are carried out using three individual GA schemes, Steady-State, Generational, and Elitism. We present new genetic operators for crossover, mutation and also an innovative encoding paradigm of CNNs to chromosomes aiming to reduce the resulting chromosome’s size by a large factor. We compare the effectiveness and scalability of our encoding with the traditional encoding. Furthermore, the performance of individual GA schemes used for training the networks were compared with each other in means of convergence rate and overall accuracy. Finally, our new encoding alongside the superior GA-based training scheme is compared to Backpropagation training with Adam optimization.

## Datasets

The code supports two image classification datasets: CIFAR-10 and MNIST. Both datasets are available in Keras and can be loaded using the `keras.datasets` module.

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The MNIST dataset consists of 70,000 28x28 grayscale images of handwritten digits, with 10 classes (0 to 9). There are 60,000 training images and 10,000 test images.

## Citation

If you use this code or data for your research, please cite the original paper as follows:

```
@article{esfahanian2019gacnn,
  title={GACNN: Training Deep Convolutional Neural Networks with Genetic Algorithm},
  author={Esfahanian, Parsa and Akhavan, Mohammad},
  journal={arXiv preprint arXiv:1909.13354},
  year={2019}
}
```
