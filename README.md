# Testing Capsule Network on various datasets

This repository contains different tests performed on a capsule network model. 

[**Test 1 : Capsule Network on mnist dataset**](#test-1-mnist---mnist)  
[**Test 2 : Capsule Network on fashion_mnist dataset**](#test-2-fashion-mnist---fashion-mnist)  
[**Test 3 : Capsule Network on small_norb dataset**](#test-3-smallnorbrandom-crop---smallnorbcenter-crop)  
[**Test 3 : Capsule Network on cifar10 dataset**](#test-4-cifar10---cifar10)  
[**Test 4 : Robustness of Capsule Network on randomly rotated mnist datset**](#test-5-mnist---mnistrotated)  
[**Test 5 : Robustness of Capsule Network on affine transformation**](#test-6-mnist---affnist)  


## Available dataset

* [mnist](http://yann.lecun.com/exdb/mnist/)
* [fashion_mnist](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)
* [affnist](http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/)
* [small_norb](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
* [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)

### data folder setting
```
-data
  -img_align_celeba
    -img1.jpg
    -img2.jpg
    -...
```
## Measurement models
* block_pixel : Each pixel is independently set to zero with probability p.  
* block_patch : A randomly chosen k × k patch is set to zero.  
* keep_patch : All pixels outside a randomly chosen k × k patch are set to zero.  
* conv_noise: k sized gaussian kernel is convolved and noise is added from the distribution Θ ∼ pθ.   

## Requirements
* python 2.7
* Tensorflow 1.4
* numpy
* cv2 (to save image tile)


## Training
```
$ python train.py --measurement=block_pixel
```

To continue training
```
$ python train.py --measurement=block_pixel --continue_training=True
```


## Block-Pixels

***Trained CelebA images***
![Alt text](images/blockpixels_train.jpg?raw=true "blockpixels celeba")
***Results***
![Alt text](images/blockpixels_result.jpg?raw=true "blockpixels result")
![Alt text](images/blockpixels_result.gif?raw=true "blockpixels result gif")


## Block-Patch

***Trained CelebA images***
![Alt text](images/blockpatch_train.jpg?raw=true "blockpatch celeba")
***Results***
![Alt text](images/blockpatch_result.jpg?raw=true "blockpatch result") 
![Alt text](images/blockpatch_result.gif?raw=true "blockpatch result gif") 

## Keep-Patch

***Trained CelebA images***
![Alt text](images/keeppatch_train.jpg?raw=true "keeppatch celeba")
***Results***
![Alt text](images/keeppatch_result.jpg?raw=true "keeppatch result")
![Alt text](images/keeppatch_result.gif?raw=true "keeppatch result gif")


## Convolve+Noise Result

## ToDo
* Test Convolve+Noise Result



