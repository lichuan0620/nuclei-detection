import os
import random
from time import strftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.morphology import label

np.random.seed()
random.seed()


def hint(message):
    """erase previous output and show new message"""
    clear_output()
    print(message)


def load_single_mask(path):
    mask = imread(path)
    return rgb2gray(mask) if len(mask.shape) is 3 else mask/255


def load_mask(path):
    """given the sample dir path, load and assemble the mask image files"""
    mask_ids = iter(next(os.walk(path + '/masks/'))[2])
    mask = load_single_mask(path + '/masks/' + next(mask_ids))
    for m_id in mask_ids:
        tmp = load_single_mask(path + '/masks/' + m_id)
        mask = np.maximum(mask, tmp)
    return mask


def load_img(path, img_id):
    """given the sample dir path, load the image with the given id"""
    return imread(path + '/images/' + img_id + '.png')


def load_images(path, img_ids, report):
    """given the sample dir path, load the image with the given ids"""
    if report:
        img_ids = tqdm(img_ids)
    return [load_img(path+img_id, img_id) for i, img_id in enumerate(img_ids)]


def load_masks(path, img_ids, report):
    """
    given the sample dir path, load and assemble the masks for the image with the given ids
    """
    if report:
        img_ids = tqdm(img_ids)
    return [load_mask(path+img_id) for img_id in img_ids]


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


def compare_images(imlist1, imlist2, sample=3, title1='', title2='', grayscale_input=True):
    """
    display some random samples with the same index number from the two list of images
    """
    _, axis = plt.subplots(sample, 2, figsize=(12, sample*6))
    samples = random.sample(range(len(imlist1)), sample)
    if grayscale_input:
        for i, sample in enumerate(samples):
            axis[i, 0].set_title(title1)
            axis[i, 0].imshow(np.squeeze(imlist1[sample], axis=-1))
            axis[i, 1].set_title(title2)
            axis[i, 1].imshow(np.squeeze(imlist2[sample], axis=-1))
    else:
        for i, sample in enumerate(samples):
            axis[i, 0].set_title(title1)
            axis[i, 0].imshow(imlist1[sample])
            axis[i, 1].set_title(title2)
            axis[i, 1].imshow(imlist2[sample])


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


def standardize_images(img_list, shape=None, inverse_color_threshold=1, dtype=np.float32):
    """
    given a list of images, return a resized version where all the images within have
    the same shape
    load the test set from the given path
    """
    to_return = [np.expand_dims(rgb2gray(img), axis=-1) for img in img_list]
    if shape is not None:
        to_return = [resize(img, shape) for img in to_return]
    if 1 > inverse_color_threshold > 0:
        for i, img in enumerate(to_return):
            if np.mean(img) > inverse_color_threshold:
                to_return[i] = 1 - img
    return np.array(to_return, dtype=dtype)


def augment_data(X, Y, vertical_flip=False, horizontal_flip=False, rotate=False, inverse_color=False):
    """
    Add augmented data to a copy of the original data
    :param X: the samples
    :param Y: the targets (the masks)
    :param vertical_flip: whether or not to flip the image vertically
    :param horizontal_flip: whether or not to flip the image horizontally
    :param rotate: whether or not to rotate the image 180
    :param inverse_color: whether or not to inverse the color of the image
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

    if inverse_color:
        assert X.shape[3] is 1
        X_ = np.append(X_, 1-X, axis=0)
        Y_ = np.append(Y_, Y, axis=0)

    return X_, Y_


def show_history(history, metric_name='acc', validation=True):
    _, axis = plt.subplots(1, 2, figsize=(12, 6))
    x = range(1, len(history[metric_name])+1)
    axis[0].plot(x, history[metric_name], label='train '+metric_name)
    axis[1].plot(x, history['loss'], label='train loss')
    if validation:
        axis[0].plot(x, history['val_'+metric_name], label='valid '+metric_name)
        axis[1].plot(x, history['val_loss'], label='valid loss')
    axis[0].grid(True)
    axis[1].grid(True)
    axis[0].set_title(metric_name)
    axis[1].set_title('loss')
    axis[0].legend()
    axis[1].legend()


def resize_prediction(original, predicted):
    """
    Resize the predicted mask to their original size
    :param original: the original images (X_)
    :param predicted: the predicted masks (Y_)
    :return: resized Y_ that matches the sizes of those in X_
    """
    return np.array([resize(y, x.shape[:2]) for x, y in zip(original, predicted)])


def format_prediction(predicted, threshold=None, original=None, reduce_dimension=False):
    """
    """
    to_return = predicted
    if threshold is not None:
        to_return = to_return > threshold
    if original is not None:
        to_return = resize_prediction(original, predicted)
    if reduce_dimension:
        to_return = np.squeeze(to_return)
    return to_return


def rle_encoding(x):
    """
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev+1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def make_submittable(ids, Y_, title='submission'):
    """
    """
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = list(prob_to_rles(Y_[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    to_return = pd.DataFrame({
        'ImageId': new_test_ids,
        'EncodedPixels': pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    })
    file_name = 'submission/' + title + '_' + strftime("%Y%m%d-%H%M%S") + '.csv'
    to_return.to_csv(file_name, index=False)
    return file_name
