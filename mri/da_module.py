# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
from collections import namedtuple
import logging
import numpy as np
from data_augmentation import Rotation, Cutout, Flip, Blur, Noise, Crop

logger = logging.getLogger()


class DAModule(object):
    def __init__(self, transforms, preproc):
        self.compose_transforms = Transformer()
        if transforms == "cutout":
            if preproc == "skeleton":
                self.compose_transforms.register(Cutout(patch_size=0.4, random_size=True, localization="on_data", 
                                                        image_shape=(128, 160, 128), min_size=0.1),
                                                probability=1)
            elif preproc == "vbm":
                self.compose_transforms.register(Cutout(patch_size=np.ceil(np.array([128]*3)/4), random_size=True, 
                                                        image_shape=(128, 128, 128)), 
                                                probability=1)
        elif transforms == "all_tf":
            if preproc == "skeleton":
                self.compose_transforms.register(Flip(), probability=0.5)
                self.compose_transforms.register(Cutout(patch_size=0.4, random_size=True, localization="on_data", 
                                                        min_size=0.1, image_shape=(128, 160, 128)),
                                                probability=1)
                self.compose_transforms.register(Crop(np.ceil(0.75*np.array([128, 160, 128])), "random", resize=True), 
                                                 probability=0.5)
            elif preproc == "vbm":
                self.compose_transforms.register(Flip(), probability=0.5)
                self.compose_transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
                self.compose_transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
                self.compose_transforms.register(Cutout(patch_size=np.ceil(np.array([128]*3)/4), random_size=True, 
                                                        image_shape=(128, 128, 128)), 
                                                probability=0.5)
                self.compose_transforms.register(Crop(np.ceil(0.75*np.array([128]*3)), "random", resize=True), 
                                                 probability=0.5)
        elif transforms == "no":
            pass
        else:
            raise ValueError(f"Unknown data augmentation : {transforms}")

    def __call__(self, x):
        return self.compose_transforms(x)

    def __str__(self):
        string = "DAModule : "
        string += self.compose_transforms.__str__()
        return string


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "params", "probability",
                                         "apply_to"])

    def __init__(self, output_label=False):
        """ Initialize the class.

        Parameters
        ----------
        output_label: bool, default False
            if output data are labels, automatically force the interpolation
            to nearest neighboor via the 'order' transform parameter.
        """
        self.transforms = []
        self.dtype = "all"
        self.output_label = output_label

    def register(self, transform, probability=1, apply_to="all", **kwargs):
        """ Register a new transformation.

        Parameters
        ----------
        transform: callable
            the transformation function.
        probability: float, default 1
            the transform is applied with the specified probability.
        apply_to: list of str, default None
            the registered transform will be only applied on specified channels
            of the input data - 'all' : all channels will be transformed
        kwargs
            the transformation function parameters.
        """
        trf = self.Transform(
            transform=transform, params=kwargs, probability=probability,
            apply_to=apply_to)
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.

        Parameters
        ----------
        arr: array
            the input data.

        Returns
        -------
        transformed: array
            the transformed input data.
        """
        return self._apply_transforms(arr)

    def _apply_transforms(self, arr):
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                if trf.apply_to == "all":
                    apply_to = [i for i in range(transformed.shape[0])]
                else:
                    apply_to = trf.apply_to
                for channel_id in apply_to:
                    transformed[channel_id] = trf.transform(
                        transformed[channel_id], **trf.params)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += f"\n\t- {trf.transform}"
        return s

    def __len__(self):
        return len(self.transforms)
