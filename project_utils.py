import os
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
    """given the sample dir path, load the image"""
    return imread(path + '/images/' + img_id + '.png')


def standardize_imgs(img_list, shape=None, grayscale=False, channel=None):
    """
    given a list of images, return a resized version where all the images within have the same shape
    """
    to_return = img_list
    assert channel is None or grayscale is None
    if channel:
        assert 5 > channel > 0
        to_return = [img[:, :, :channel] for img in to_return]
    if grayscale:
        to_return = [rgb2gray(img) for img in to_return]
    if shape is not None:
        to_return = [resize(img, shape, mode='reflect') for img in to_return]
    return to_return
