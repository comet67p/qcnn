from skimage.filters import gabor_kernel
import numpy as np
from scipy import ndimage as nd
import math 

from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def compute_feats(img, kernel):
  feats = np.zeros((2), dtype=np.double)
  filtered = nd.convolve(img, kernel, mode='wrap')
  feats[0] = filtered.mean()
  feats[1] = filtered.var()
  if(math.isnan(feats[0])):
    feats[0] = 0.
  if(math.isnan(feats[1])):
    feats[1] = 0.
  return feats

def compute_power_feats(image, kernel):
  power_img = power(image, kernel)
  power_feats = np.zeros((2), dtype=np.double)
  power_feats[0] = power_img.mean()
  power_feats[1] = power_img.var()
  if(math.isnan(power_feats[0])):
    power_feats[0] = 0.
  if(math.isnan(power_feats[1])):
    power_feats[1] = 0.
  return (power_feats, power_img)

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
                   nd.convolve(image, np.imag(kernel), mode='wrap')**2)

def extract_feats(data, kernelSize, i, j, frequency, theta, sigma_x=1, sigma_y=1):
  img = data[0, 0, i:i+kernelSize, j:j+kernelSize].numpy().squeeze()
  kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y))
  feats = compute_feats(img, kernel)
  return (kernel, feats)

def extract_power_feats(data, kernelSize, i, j, frequency, theta, sigma_x=1, sigma_y=1):
  img = data[0, 0, i:i+kernelSize, j:j+kernelSize].numpy().squeeze()
  kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y))
  power_feats, power_img = compute_power_feats(img, kernel)
  return (kernel, power_feats, power_img)


def extract_power_featsV2(data, kernelSize, i, j, frequency, theta, sigma_x=1, sigma_y=1):
  img = data[0, i:i+kernelSize, j:j+kernelSize, 0].numpy().squeeze()
  kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y))
  power_feats, power_img = compute_power_feats(img, kernel)
  # print("data", data.shape, "img", img.shape, "feat", power_feats, power_img)
  return (kernel, power_feats, power_img)

def extract_all(img, frequency, theta, sigma_x=1, sigma_y=1):
  kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma_x, sigma_y=sigma_y))
  feats = compute_feats(img, kernel)
  power_feats, power_img = compute_power_feats(img, kernel)
  return (kernel, feats, power_feats, power_img)


def demo(): 
  ####################################33
  # Use pre-defined torchvision function to load MNIST train data
  X_train = datasets.MNIST(
      root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
  )

  batch_size = 1
  n_samples = 10
  # Filter out labels (originally 0-9), leaving only labels 0 and 1
  idx = np.where(X_train.targets < 3)[0][:n_samples]
  X_train.data = X_train.data[idx]
  X_train.targets = X_train.targets[idx]
  train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
  data_iter = iter(train_loader)

  images, targets = data_iter._next_data()
  img1 = images[0, 0].numpy().squeeze()
  imgName1 = targets[0].item()


  fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
  plt.gray()
  fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
  axes[0][0].axis('off')

  for img, ax in zip(images, axes[0][1:]):
      ax.imshow(img.squeeze())
      ax.axis('off')

  kernel, feats, power_feats, power_img = extract_all(img1, 0.1, 1/4. * np.pi)


  ax = axes[1,0]
  ax.imshow(np.real(kernel), interpolation='nearest')
  ax.set_xticks([])
  ax.set_yticks([])

  print("feat", feats, "power feats", power_feats)

  ax = axes[2,0]
  ax.imshow(np.real(kernel), interpolation='nearest')
  ax.set_xticks([])
  ax.set_yticks([])

  ax = axes[2,1]
  vmin = np.min(power_img)
  vmax = np.max(power_img)
  ax.imshow(power_img, vmin=vmin, vmax=vmax)
  ax.axis('off')

  plt.show()


# demo()