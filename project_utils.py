import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


def hint(message):
    """erase previous output and show new message"""
    clear_output()
    print(message)


def load_mask(path):
    """given the sample dir path, load and assemble the mask image files"""
    mask_ids = iter(next(os.walk(path + '/masks/'))[2])
    mask = imread(path + '/masks/' + next(mask_ids))
    for m_id in mask_ids:
        mask += imread(path + '/masks/' + m_id)
    return mask


def load_img(path, img_id):
    """given the sample dir path, load the image with the given id"""
    return imread(path + '/images/' + img_id + '.png')


def load_images(path, img_ids, report=True):
    """given the sample dir path, load the image with the given ids"""
    if report:
        img_ids = tqdm(img_ids)
    return np.array([load_img(path+img_id, img_id) for i, img_id in enumerate(img_ids)])


def load_masks(path, img_ids, report=True):
    """
    given the sample dir path, load and assemble the masks for the image with the given ids
    """
    if report:
        img_ids = tqdm(img_ids)
    return np.array([load_mask(path+img_id) for img_id in img_ids])


def standardize_images(img_list, shape=None, grayscale=False):
    """
    given a list of images, return a resized version where all the images within have
    the same shape
    """
    to_return = img_list
    if grayscale:
        to_return = [rgb2gray(img) for img in to_return]
        # to_return = [np.expand_dims(rgb2gray(img), axis=-1) for img in to_return]
    if shape is not None:
        to_return = [resize(img, shape, mode='reflect') for img in to_return]
    return np.array(to_return)


def compare_images(imlist1, imlist2, sample=3, title1='', title2=''):
    """
    select the given number of pairs of images from the two list and display them
    side by side
    """
    total_sample = imlist1.shape[0]
    assert total_sample == imlist2.shape[0] and total_sample >= sample
    fig, axis = plt.subplots(sample, 2, figsize=(12, sample * 6))
    samples = random.sample(range(len(imlist1)), sample)
    for i, sample in enumerate(samples):
        axis[i, 0].imshow(imlist1[sample])
        axis[i, 0].set_title(title1)
        axis[i, 1].imshow(imlist2[sample])
        axis[i, 1].set_title(title2)


def augment_data(X, Y, vertical_flip=False, horizontal_flip=False, rotate=False):
    """
    Add augmented data to a copy of the original data
    :param X: the samples
    :param Y: the targets (the masks)
    :param vertical_flip: whether or not to flip the image vertically
    :param horizontal_flip: whether or not to flip the image horizontally
    :param rotate: whether or not to rotate the image 180
    :return: a new array containing the original and the augmented data
    """
    X_, Y_ = X, Y

    if vertical_flip:
        X_ = np.append(X_, np.flip(X, axis=1), axis=0)
        Y_ = np.append(Y_, np.flip(Y, axis=1), axis=0)

    if horizontal_flip:
        X_ = np.append(X_, np.flip(X, axis=2), axis=0)
        Y_ = np.append(Y_, np.flip(Y, axis=2), axis=0)

    if rotate:
        X_ = np.append(X_, np.flip(np.flip(X, axis=1), axis=2), axis=0)
        Y_ = np.append(Y_, np.flip(np.flip(Y, axis=1), axis=2), axis=0)

    return X_, Y_
