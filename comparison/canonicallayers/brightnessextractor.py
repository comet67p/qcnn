import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss, Softmax
from torch.optim import LBFGS
import warnings

import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.nn.functional as F

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

KERNEL_SIZE = 5

def extractBrightnessAtV2(data, tuneTo, i, j, kernelSize):
    image = data[0]
    imagesize=list(image.shape)[0]
    kernel= image[i:i+kernelSize, j:j+kernelSize, :]
    sum = kernel.sum() / (kernelSize*kernelSize)
    value = np.abs(1/ (tuneTo - sum))
    return value

def extractBrightnessAt(data, tuneTo, i, j, kernelSize):
    image = data[0,0]
    imagesize=list(image.shape)[0]
    kernel= image[i:i+kernelSize, j:j+kernelSize]
    sum = kernel.sum() / (kernelSize*kernelSize)
    value = 1 - np.abs(tuneTo - sum)
    w = 2 * value - 1
    return w

def extractBrightness(data, tuneTo):
    image = data[0,0]
    imagesize=list(image.shape)[0]
    # print("shape of image", image.shape)
    w = torch.zeros([imagesize - KERNEL_SIZE + 1, imagesize - KERNEL_SIZE + 1])
    for i in range(0, imagesize - KERNEL_SIZE +1):
        for j in range(0, imagesize - KERNEL_SIZE +1):
            w[i, j] = extractBrightnessAt(data, tuneTo, i, j, KERNEL_SIZE)
                # print("Weights", w)
    return w
