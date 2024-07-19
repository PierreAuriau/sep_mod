# !/usr/bin/env python
# -*- coding: utf-8 -*-


# Import
import logging
import numpy as np
from scipy.ndimage import rotate, gaussian_filter, shift
from skimage import transform as sk_tf
import torch
import torch.nn as nn
import numbers

logger = logging.getLogger()

class ToTensor(nn.Module):
    def forward(self, arr):
        return torch.from_numpy(arr)


class ToArray(nn.Module):
    def forward(self, tensor):
        return np.asarray(tensor)

def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")

    return tuple(obj)   


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean
    
    def __str__(self):
        return "Normalize"


class Rotation(object):

    def __init__(self, angles, axes=(0, 2), reshape=False, **kwargs):
        if isinstance(angles, (int, float)):
            self.angles = [-angles, angles]
        elif isinstance(angles, (list, tuple)):
            assert (len(angles) == 2 and angles[0] < angles[1]), print(f"Wrong angles format {angles}")
            self.angles = angles
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        if isinstance(axes, tuple):
            self.axes = [axes]
        elif isinstance(axes, list):
            self.axes = axes
        else:
            logger.warning('Rotations: rotation plane will be determined randomly')
            self.axes = [tuple(np.random.choice(3, 2, replace=False))]
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        return self._apply_random_rotation(arr)

    def _apply_random_rotation(self, arr):
        angles = [np.float16(np.random.uniform(self.angles[0], self.angles[1]))
                  for _ in range(len(self.axes))]
        for ax, angle in zip(self.axes, angles):
            arr = rotate(arr, angle, axes=ax, reshape=self.reshape, **self.rotate_kwargs)
        return arr

    def __str__(self):
        return f"Rotation(angles={self.angles}, axes={self.axes})"


class Cutout(object):
    def __init__(self, patch_size, random_size=False, localization="random", **kwargs):
        self.patch_size = patch_size
        self.random_size = random_size
        if localization in ["random", "on_data"] or isinstance(localization, (tuple, list)):
            self.localization = localization
        else:
            logger.warning("Cutout : localization is set to random")
            self.localization = "random"
        self.min_size = kwargs.get("min_size", 0)
        self.value = kwargs.get("value", 0)
        self.image_shape = kwargs.get("image_shape", None)
        if self.image_shape is not None:
            self.patch_size = self._get_patch_size(self.patch_size,
                                                   self.image_shape)
            self.min_size = self._get_patch_size(self.min_size,
                                                 self.image_shape)
        self.shuffle = kwargs.get("shuffle", False)
        if self.shuffle:
            logger.warning("Cutout: shuffle pixels, ignoring value")

    def __call__(self, arr):
        if self.image_shape is None:
            arr_shape = arr.shape
            self.patch_size = self._get_patch_size(self.patch_size,
                                                   arr_shape)
            self.min_size = self._get_patch_size(self.min_size,
                                                 arr_shape)
        return self._apply_cutout(arr)

    def _apply_cutout(self, arr):
        image_shape = arr.shape
        if self.localization == "on_data":
            nonzero_voxels = np.nonzero(arr)
            index = np.random.randint(0, len(nonzero_voxels[0]))
            localization = np.array([nonzero_voxels[i][index] for i in range(len(nonzero_voxels))])
        elif isinstance(self.localization, (tuple, list)):
            assert len(self.localization) == len(image_shape), f"Cutout : wrong localization shape"
            localization = self.localization
        else:
            localization = None
        indexes = []
        for ndim, shape in enumerate(image_shape):
            if self.random_size:
                size = np.random.randint(self.min_size[ndim], self.patch_size[ndim])
            else:
                size = self.patch_size[ndim]
            if localization is not None:
                delta_before = max(localization[ndim] - size // 2, 0)
            else:
                delta_before = np.random.randint(0, shape - size + 1)
            indexes.append(slice(delta_before, delta_before + size))
        if self.shuffle:
            sh = [s.stop - s.start for s in indexes]
            arr[tuple(indexes)] = np.random.shuffle(arr[tuple(indexes)].flat).reshape(sh)
        else:
            arr[tuple(indexes)] = self.value
        return arr

    @staticmethod
    def _get_patch_size(patch_size, image_shape):
        if isinstance(patch_size, int):
            size = [patch_size for _ in range(len(image_shape))]
        elif isinstance(patch_size, float):
            size = [int(patch_size*s) for s in image_shape]
        else:
            size = patch_size
        assert len(size) == len(image_shape), "Incorrect patch dimension."
        for ndim in range(len(image_shape)):
            if size[ndim] > image_shape[ndim] or size[ndim] < 0:
                size[ndim] = image_shape[ndim]
        return size

    def __str__(self):
        return f"Cutout(patch_size={self.patch_size}, random_size={self.random_size}, " \
               f"localization={self.localization})"
    

class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""
    def __init__(self, shape, localization="center", resize=False, keep_dim=False, random_size=False, image_shape=None):
        """:param
        shape: int, float, tuple or list of int
            The shape of the patch to crop
        localization: 'center' or 'random'
            Whether the crop will be centered or at a random location
        resize: bool, default False
            If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
        keep_dim: bool, default False
            if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
        """
        assert localization in ["center", "random"], f"Unkwnown localization {localization}"
        self.shape = shape
        self.localization = localization
        self.random_size = random_size
        self.resize=resize
        self.keep_dim=keep_dim
        if image_shape is not None:
            self.image_shape = image_shape
            self.crop_size = self._set_crop_size(shape=shape, image_shape=image_shape)
        else:
            self.image_shape = None

    def __call__(self, arr):
        if self.image_shape is None:
            self.image_shape = arr.shape
            self.crop_size = self._get_crop_size(shape=self.shape, image_shape=self.image_shape)
        indexes = []
        for ndim, img_shape in enumerate(self.image_shape):
            if self.random_size:
                size = np.random.randint(self.crop_size[ndim], img_shape)
            else:
                size = self.crop_size[ndim]
            if self.localization == "center":
                delta_before = (img_shape - size) / 2.0
            elif self.localization == "random":
                delta_before = np.random.randint(0, img_shape - size + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size)))
        if self.resize:
            # resize the image to the input shape
            return sk_tf.resize(arr[tuple(indexes)], self.image_shape, preserve_range=True)

        if self.keep_dim:
            mask = np.zeros(self.image_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = arr.copy()
            arr_copy[~mask] = 0
            return arr_copy

        return arr[tuple(indexes)]
    
    @staticmethod
    def _get_crop_size(shape, image_shape):
        if isinstance(shape, int):
            size = [shape for _ in range(len(image_shape))]
        elif isinstance(shape, float):
            size = [int(shape*s) for s in image_shape]
        else:
            size = shape
        assert len(size) == len(image_shape), f"Shape of array {image_shape} does not match {shape}"
        for ndim in range(len(image_shape)):
            if size[ndim] > image_shape[ndim] or size[ndim] < 0:
                size[ndim] = image_shape[ndim]
        return size

    def __str__(self):
        return f"Crop(shape={self.shape}, random_size={self.random_size}, " \
               f"localization={self.localization})"
    
class Flip(object):
    """ Apply a random mirror flip."""
    def __init__(self, axis=None):
        '''
        :param axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        '''
        self.axis = axis

    def __call__(self, arr):
        if self.axis is None:
            axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
        return np.flip(arr, axis=(self.axis or axis))
    
    def __str__(self):
        return f"Flip(axis={self.axis})"


class Blur(object):
    def __init__(self, snr=None, sigma=None):
        """ Add random blur using a Gaussian filter.
            Parameters
            ----------
            snr: float, default None
                the desired signal-to noise ratio used to infer the standard deviation
                for the noise distribution.
            sigma: float or 2-uplet
                the standard deviation for Gaussian kernel.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        self.snr = snr
        self.sigma = sigma

    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        return gaussian_filter(arr, sigma_random)
    
    def __str__(self):
        return f"Blur(snr={self.snr}, sigma={self.sigma})"


class Noise(object):
    def __init__(self, snr=None, sigma=None, noise_type="gaussian"):
        """ Add random Gaussian or Rician noise.

           The noise level can be specified directly by setting the standard
           deviation or the desired signal-to-noise ratio for the Gaussian
           distribution. In the case of Rician noise sigma is the standard deviation
           of the two Gaussian distributions forming the real and imaginary
           components of the Rician noise distribution.

           In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
           3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).

           Parameters
           ----------
           snr: float, default None
               the desired signal-to noise ratio used to infer the standard deviation
               for the noise distribution.
           sigma: float or 2-uplet, default None
               the standard deviation for the noise distribution.
           noise_type: str, default 'gaussian'
               the distribution of added noise - can be either 'gaussian' for
               Gaussian distributed noise, or 'rician' for Rice-distributed noise.
        """

        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        assert noise_type in {"gaussian", "rician"}, "Noise muse be either Rician or Gaussian"
        self.snr = snr
        self.sigma = sigma
        self.noise_type = noise_type


    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        noise = np.random.normal(0, sigma_random, [2] + list(arr.shape))
        if self.noise_type == "gaussian":
            transformed = arr + noise[0]
        elif self.noise_type == "rician":
            transformed = np.square(arr + noise[0])
            transformed += np.square(noise[1])
            transformed = np.sqrt(transformed)
        return transformed
    
    def __str__(self):
        return f"Noise(snr={self.snr})"
    

class Shift(object):
    """ Translate the image of a number of voxels.
    """
    def __init__(self, nb_voxels, random):
        self.random = random
        self.nb_voxels = nb_voxels

    def __call__(self, arr):
        ndim = arr.ndim
        if self.random:
            translation = np.random.randint(-self.nb_voxels, self.nb_voxels+1, size=ndim)
        else:
            if isinstance(self.nb_voxels, int):
                translation = [self.nb_voxels for _ in range(ndim)]
            else:
                translation = self.nb_voxels
        transformed = shift(arr, translation, order=0, mode='constant', cval=0.0, prefilter=False)
        return transformed