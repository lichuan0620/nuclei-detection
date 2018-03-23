import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

np.random.seed()
random.seed()


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


def load_images(path, img_ids, report):
    """given the sample dir path, load the image with the given ids"""
    if report:
        img_ids = tqdm(img_ids)
    return np.array([load_img(path+img_id, img_id) for i, img_id in enumerate(img_ids)])


def load_masks(path, img_ids, report):
    """
    given the sample dir path, load and assemble the masks for the image with the given ids
    """
    if report:
        img_ids = tqdm(img_ids)
    return np.array([load_mask(path+img_id) for img_id in img_ids])


def load_train_set(path, report=True):
    """
    load the train set from the given path
    """
    to_return = pd.DataFrame(columns=['id', 'image', 'mask'])
    to_return['id'] = next(os.walk(path))[1]
    if report:
        print("Loading images...")
    to_return['image'] = load_images(path, to_return['id'], report)
    if report:
        print("Loading masks...")
    to_return['mask'] = load_masks(path, to_return['id'], report)
    return to_return


def load_test_set(path, report=True):
    """
    load the test set from the given path
    """
    to_return = pd.DataFrame(columns=['id', 'image'])
    to_return['id'] = next(os.walk(path))[1]
    if report:
        print("Loading images...")
    to_return['image'] = load_images(path, to_return['id'], report)
    return to_return


def compare_images(imlist1, imlist2, sample=3, title1='', title2=''):
    """
    display some random samples with the same index number from the two list of images
    """
    _, axis = plt.subplots(sample, 2, figsize=(12, sample*6))
    samples = random.sample(range(len(imlist1)), sample)
    for i, sample in enumerate(samples):
        axis[i, 0].imshow(imlist1[sample])
        axis[i, 0].set_title(title1)
        axis[i, 1].imshow(imlist2[sample])
        axis[i, 1].set_title(title2)


def get_dim_stat(images):
    """
    gather dimension information of the images
    """
    to_return = np.zeros((len(images), 4))
    for i, img in enumerate(images):
        to_return[i, 0] = img.shape[0]
        to_return[i, 1] = img.shape[1]
        to_return[i, 2] = img.shape[0]/img.shape[1]
        to_return[i, 3] = img.shape[2]

    return pd.DataFrame(
        to_return,
        columns=['height', 'width', 'h/w ratio', 'channel count']
    )


def standardize_images(img_list, shape=None, grayscale=False):
    """
    given a list of images, return a resized version where all the images within have
    the same shape
    load the test set from the given path
    """
    to_return = [rgb2gray(img) if grayscale else img[:, :, :3] for img in img_list]
    if shape is not None:
        to_return = [resize(img, shape, mode='reflect') for img in to_return]
    return np.array(to_return)


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
