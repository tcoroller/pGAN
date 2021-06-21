import numpy as np
from experiments.pgan.transforms import augmentations

class RotationFixedAxis:
    """Rotate randomly the 3D image around a given axis

    Args:
        max_angle (float or int): Random rotation angle (in degree), drawn as uniform between [-max_angle, max_angle]
        axis (int): Dimension of the array used as axis of rotation axis is 0, 1 or 2
        reshape (bool): Set reshape argument so the output rotated image is extended and no data is cropped out,
            default is reshape=False"""

    def __init__(self, max_angle=15.0, axis=0, reshape=False):
        self.axis = axis
        self.rand_angle = np.random.uniform(-np.abs(max_angle), np.abs(max_angle))
        self.reshape = reshape
        assert axis >= 0 & axis < 3
        assert isinstance(reshape, bool)
        assert isinstance(max_angle, (float, int))

    def __call__(self, sample):
        rotated_image = augmentations.rotate_fixed_axis(sample, angle=self.rand_angle,
                                                        axis=self.axis, reshape=self.reshape)
        return rotated_image

class RotationRandomAxis:
    """Rotate randomly the 3D image around a random axis

    Args:
        max_angle (float or int): Random rotation angle (in degree), drawn as uniform between [-max_angle, max_angle]
        reshape (bool): Set reshape argument so the output rotated image is extended and no data is cropped out,
            default is reshape=False"""

    def __init__(self, max_angle=15.0, reshape=False):
        self.rand_angle = np.random.uniform(-np.abs(max_angle), np.abs(max_angle))
        self.reshape = reshape
        assert isinstance(reshape, bool)
        assert isinstance(max_angle, (float, int))

    def __call__(self, sample):
        rotated_image = augmentations.rotate_random_axis(sample, angle=self.rand_angle,
                                                         reshape=self.reshape)
        return rotated_image

class Rotation3D:
    """Crop randomly the image in a sample.

    Args:
        max_angle (float or tuple): Random rotation angle (in degree), drawn as uniform between [-max_angle, max_angle]
            (float) each component within the same range
            (tuple) to specify each angle (max_angle_x, max_angle_y, max_angle_z)
        reshape (bool): Set reshape argument so the output rotated image is extended and no data is cropped out,
            default is reshape=False"""

    def __init__(self, max_angle=15.0, reshape=False):
        assert isinstance(max_angle, (int, float, tuple, np.ndarray, list))
        assert isinstance(reshape, bool)
        self.reshape = reshape

        if isinstance(max_angle, (int, float)):
            self.max_angle = (max_angle, max_angle, max_angle)
        else:
            assert len(max_angle) == 3
            self.max_angle = max_angle

        self.rand_angle = [np.random.uniform(-np.abs(self.max_angle[0]), np.abs(self.max_angle[0])),
                           np.random.uniform(-np.abs(self.max_angle[1]), np.abs(self.max_angle[1])),
                           np.random.uniform(-np.abs(self.max_angle[2]), np.abs(self.max_angle[2]))]

    def __call__(self, sample):
        rotated_image = augmentations.rotate_fixed_axis(sample, angle=self.rand_angle[0],
                                                        axis=0, reshape=self.reshape)
        rotated_image = augmentations.rotate_fixed_axis(rotated_image, angle=self.rand_angle[1],
                                                        axis=1, reshape=self.reshape)
        rotated_image = augmentations.rotate_fixed_axis(rotated_image, angle=self.rand_angle[2],
                                                        axis=2, reshape=self.reshape)
        return rotated_image

class Rescale:
    """Rescale the image on 2 axis

    Args:
        ratio (float or tuple): multiplying factor by which each axis is multiplied
            (float) ratio is the same for each axis
            (tuple) ratio is (ratio_x, ratio_y)
        axis (int): Dimensions for rescaling
            (axis=0) x,y
            (axis=1) x,z
            (axis=2) z,y"""

    def __init__(self, ratio, axis=0):
        assert isinstance(ratio, (int, float, tuple, np.ndarray, list))
        assert isinstance(axis, int)
        self.axis = axis
        if isinstance(ratio, (int, float)):
            self.ratio = (ratio, ratio, ratio)
        else:
            assert len(ratio) == 2
            self.ratio = ratio

    def __call__(self, sample):
        rescaled_image = augmentations.rescale(sample, self.ratio[0], self.ratio[1], axis=self.axis)
        return rescaled_image

class CenterCrop:
    """Crop around the center of the image.

    Args:
        output_size (int or tuple): crop size
            (int) all components are cropped to be of size output_size
            (tuple) components are cropped according to (output_size_x, output_size_y, output_size_z)"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, np.ndarray, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        cropped_image = augmentations.center_crop(sample, self.output_size[0],
                                                  self.output_size[1], self.output_size[2])
        return cropped_image

class RandomCrop:
    """Crop the image randomly

    Args:
        output_size (int or tuple): crop size
            (int) all components are cropped to be of size output_size
            (tuple) components are cropped according to (output_size_x, output_size_y, output_size_z)"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, np.ndarray, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        cropped_image = augmentations.random_crop(sample, self.output_size[0],
                                                  self.output_size[1], self.output_size[2])
        return cropped_image

class Flip:
    """Flips the image across a given axis

    Args:
        axis (int): Dimension of the array used as axis for flip, axis can be 0, 1 or 2
        flip_probability (float): probability of flip, default is 0.5"""

    def __init__(self, axis=0, flip_probability=0.5):
        assert isinstance(axis, int)
        assert isinstance(flip_probability, (int, float))
        self.axis = axis
        self.flip_probability = flip_probability

    def __call__(self, sample):
        flipped_image = augmentations.flip(sample, axis=self.axis, probability=self.flip_probability)
        return flipped_image

class Whitening:
    """Homogenous modification of the intensity of the image by shifting by a random proportion.
    im_augmented = im_base + a*im_base
    Where a is randomly generated

    Args:
        min_add_intensity (float): min additional values for shifting intensity (default=-0.1)
        max_add_intensity (float): max additional values for shifting intensity (default=0.1)"""

    def __init__(self, min_add_intensity=-0.1, max_add_intensity=0.1):
        assert isinstance(min_add_intensity, (int, float))
        assert isinstance(max_add_intensity, (int, float))
        self.min_add_intensity = min_add_intensity
        self.max_add_intensity = max_add_intensity

    def __call__(self, sample):
        add_intensity = np.random.uniform(self.min_add_intensity, self.max_add_intensity)
        whitened_image = augmentations.uniform_intensity_additive(sample, additional_intensity=add_intensity)
        return whitened_image

class Zoom:
    """Zoom image using scipy ndimage. Zoom is applied on component 2 and 3 that are scaled by the same value

    Args:
        max_zoom (float): Zoom proportion (if max_zoom = 1.15) images gets 1.15*original size
        random (bool): if true, sample random zoom between [1, max_zoom],
            otherwise always zoom to max_zoom (default: False)"""

    def __init__(self, max_zoom, random=False):
        assert isinstance(max_zoom, (int, float))
        assert isinstance(random, bool)
        self.max_zoom = max_zoom
        self.random = random

    def __call__(self, sample):
        if self.random:
            zoom_f = np.random.uniform(1, self.max_zoom)
        else:
            zoom_f = self.max_zoom
        zoomed_image = augmentations.zoom(sample, zoom_factor=zoom_f)
        return zoomed_image
