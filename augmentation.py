import torch
import random
import torchvision
import numpy as np
import imgaug.augmenters as iaa

hflip = torchvision.transforms.functional.hflip # hflip 
vflip = torchvision.transforms.functional.vflip # vflip
affine = torchvision.transforms.functional.affine # affine
rotate = torchvision.transforms.functional.rotate # rotate
add = iaa.Add((-10, 10)) # add
multiply = iaa.Multiply((0.9, 1.1)) # multifly

def policy1(img): # carpet, tile, leather, grid 
  augmentations = [hflip, vflip, affine, rotate, add, multiply]  
  aug_name = ['hflip', 'vfilp', 'affine', 'rotate', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True) # random choice 1~maxnum augmentation
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=0, translate=(torch.randint(low=-5, high=5, size=(1,)).item(), torch.randint(low=-5, high=5, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] == 'rotate':
      img = augmentations[ind](img, 90)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy2(img): # capsule, pill 
  augmentations = [affine, add, multiply]  
  aug_name = ['affine', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=(torch.randint(low=-3, high=3, size=(1,)).item()), translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-10, high=10, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy3(img): # cable
  augmentations = [affine, add, multiply]  
  aug_name = ['affine', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=(torch.randint(low=-5, high=5, size=(1,)).item()), translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-10, high=10, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy4(img): # transistor
  augmentations = [hflip, affine, add, multiply]  
  aug_name = ['hflip', 'affine', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=(torch.randint(low=-2, high=2, size=(1,)).item()), translate=(torch.randint(low=-5, high=5, size=(1,)).item(), torch.randint(low=-5, high=5, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy5(img): # metal nut
  augmentations = [affine, rotate, add, multiply]  
  aug_name = ['affine', 'rotate', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=(torch.randint(low=-10, high=10, size=(1,)).item()), translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-10, high=10, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] == 'rotate': 
      img = augmentations[ind](img, 90)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy6(img): # toothbrush
  augmentations = [hflip, affine, add, multiply]  
  aug_name = ['hflip', 'affine', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=0, translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-10, high=10, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy7(img): # screw
  augmentations = [hflip, vflip, affine, rotate, add, multiply]  
  aug_name = ['hflip', 'vflip', 'affine', 'rotate', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=(torch.randint(low=-10, high=10, size=(1,)).item()), translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-10, high=10, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] == 'rotate': 
      img = augmentations[ind](img, 90)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img

def policy8(img): # hazelnut
  augmentations = [hflip, vflip, affine, rotate, add, multiply]  
  aug_name = ['hflip', 'vflip', 'affine', 'rotate', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=torch.randint(low=-20, high=20, size=(1, )).item(), translate=(torch.randint(low=-10, high=10, size=(1,)).item(), torch.randint(low=-5, high=5, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] == 'rotate': 
      img = augmentations[ind](img, 90)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img 

def policy9(img): # zipper 
  augmentations = [hflip, affine, add, multiply]  
  aug_name = ['hflip', 'affine', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=0, translate=(torch.randint(low=-30, high=30, size=(1,)).item(), torch.randint(low=0, high=1, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img 

def policy10(img): #bottle
  augmentations = [affine, rotate, add, multiply]  
  aug_name = ['affine', 'rotate', 'add', 'multiply']  
  aug_ind = np.arange(len(augmentations))
  aug_sets = np.random.choice(aug_ind, size=random.randint(1, len(augmentations)), replace=True)
  for ind in aug_sets:
    if aug_name[ind] == 'affine':
      img = augmentations[ind](img, angle=torch.randint(low=-10, high=10, size=(1,)).item(), translate=(torch.randint(low=-5, high=5, size=(1,)).item(), torch.randint(low=-5, high=5, size=(1,)).item()), scale=1, shear=0)
    elif aug_name[ind] == 'rotate':
      img = augmentations[ind](img, 90)
    elif aug_name[ind] in ['add', 'multiply']:
      img = augmentations[ind].augment_image(np.array(img))
    else:
      img = augmentations[ind](img)
    return img
