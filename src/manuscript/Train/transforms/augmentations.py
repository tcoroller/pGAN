import numpy as np
from scipy import interpolate
from scipy import ndimage

def rotate_fixed_axis(image, angle=10, axis=0, reshape=False):

    if axis == 0:
        pass
    elif axis == 1:
        image = image.transpose((1, 0, 2))
    elif axis == 2:
        image = image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    list_rotate = [0]*len(image)
    for idx, im_slice in enumerate(image):
        list_rotate[idx] = ndimage.rotate(im_slice, angle=angle, axes=(0, 0), reshape=reshape)

    rotated_image = np.array(list_rotate)

    if axis == 0:
        pass
    elif axis == 1:
        rotated_image = rotated_image.transpose((1, 0, 2))
    elif axis == 2:
        rotated_image = rotated_image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    return rotated_image

def rotate_random_axis(image, angle=10, reshape=False):
    axis = np.random.randint(3)

    if axis == 0:
        pass
    elif axis == 1:
        image = image.transpose((1, 0, 2))
    elif axis == 2:
        image = image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    list_rotate = [0]*len(image)
    for idx, im_slice in enumerate(image):
        list_rotate[idx] = ndimage.rotate(im_slice, angle=angle, axes=(0, 0), reshape=reshape)

    rotated_image = np.array(list_rotate)

    if axis == 0:
        pass
    elif axis == 1:
        rotated_image = rotated_image.transpose((1, 0, 2))
    elif axis == 2:
        rotated_image = rotated_image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    return rotated_image

def rescale(image, ratio_x, ratio_y, axis=0):

    if axis == 0:
        pass
    elif axis == 1:
        image = image.transpose((1, 0, 2))
    elif axis == 2:
        image = image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    im_shape = image[0].shape

    temp_init_grid = np.mgrid[0:im_shape[0]:1, 0:im_shape[1]:1].T.reshape(-1, 2)

    eps = 1e-4
    temp_final_grid = np.mgrid[0:im_shape[0]-1+eps:(im_shape[0]-1)/(im_shape[0]*ratio_x - 1),
                               0:im_shape[1]-1+eps:(im_shape[1]-1)/(im_shape[1]*ratio_y - 1)].T.reshape(-1, 2)

    list_rescale = [0]*len(image)

    for idx, im_slice in enumerate(image):
        new_im = interpolate.griddata(temp_init_grid, im_slice.ravel(), temp_final_grid, method='linear')
        list_rescale[idx] = new_im.reshape(int(im_shape[1]*ratio_y), int(im_shape[0]*ratio_x))

    rescaled_image = np.array(list_rescale)

    if axis == 0:
        pass
    elif axis == 1:
        rescaled_image = rescaled_image.transpose((1, 0, 2))
    elif axis == 2:
        rescaled_image = rescaled_image.transpose((2, 1, 0))
    else:
        print('axis out of bounds')

    return rescaled_image

def center_crop(image, size_x, size_y, size_z):
    im_shape = np.array(image.shape, dtype=np.int)
    im_center = np.array(im_shape/2, dtype=np.int)

    assert size_x <= im_shape[0]
    assert size_y <= im_shape[1]
    assert size_z <= im_shape[2]

    st_x = int(np.floor(im_center[0]))-int(np.floor(size_x/2))
    end_x = int(np.ceil(im_center[0]))+int(np.ceil(size_x/2))

    st_y = int(np.floor(im_center[1]))-int(np.floor(size_y/2))
    end_y = int(np.ceil(im_center[1]))+int(np.ceil(size_y/2))

    st_z = int(np.floor(im_center[2]))-int(np.floor(size_z/2))
    end_z = int(np.ceil(im_center[2]))+int(np.ceil(size_z/2))
    im_crop = image[st_x:end_x, st_y:end_y, st_z:end_z]
    return im_crop

def random_crop(image, size_x, size_y, size_z):
    im_shape = np.array(image.shape, dtype=np.int)

    assert size_x <= im_shape[0]
    assert size_y <= im_shape[1]
    assert size_z <= im_shape[2]

    range_x = im_shape[0] - size_x + 1
    rand_st_x = np.random.randint(range_x)
    end_x = rand_st_x + size_x

    range_y = im_shape[1] - size_y + 1
    rand_st_y = np.random.randint(range_y)
    end_y = rand_st_y + size_y

    range_z = im_shape[2] - size_z + 1
    rand_st_z = np.random.randint(range_z)
    end_z = rand_st_z + size_z

    im_crop = image[rand_st_x:end_x, rand_st_y:end_y, rand_st_z:end_z]
    return im_crop

def flip(image, axis=0, probability=0.5):
    if np.random.uniform() < probability:
        return np.flip(image, axis=axis)
    else:
        return image

def zoom(image, zoom_factor=1.10):
    layer1 = ndimage.zoom(image[0, :, :], zoom_factor)
    temp = np.zeros([image.shape[0], layer1.shape[0], layer1.shape[1]])
    for i in range(image.shape[0]):
        temp[i, :, :] = ndimage.zoom(image[i, :, :], zoom_factor)

    return temp

def uniform_intensity_additive(image, additional_intensity=0.1):
    return image + additional_intensity
