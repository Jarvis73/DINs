"""
Some utility functions run in tensorflow graph
"""

import tensorflow as tf
import tensorflow_addons as tfa


def z_score(img):
    """
    Z-score image normalization. Support for N-D volumes.

    Parameters
    ----------
    img: tf.Tensor, normalization is performed on all the axes

    Returns
    -------
    new_img: Normalized image
    """
    nonzero_region = img > 0
    flatten_img = tf.reshape(img, [-1])
    flatten_mask = tf.reshape(nonzero_region, [-1])
    mean, variance = tf.nn.moments(tf.boolean_mask(flatten_img, flatten_mask), axes=(0,))
    float_region = tf.cast(nonzero_region, img.dtype)
    new_img = (img - float_region * mean) / (float_region * tf.math.sqrt(variance) + 1e-8)
    return new_img


def random_flip(image, label=None, seed=None, flip=1, name="random_flip", extra=None):
    """Randomly flip an image horizontally (left to right), vertically (up to down)
    or (front to back).

    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    specified dimension.  Otherwise output the image as-is.

    Parameters
    ----------
    image: 4-D Tensor of shape `[depth, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[depth, height, width]` or
           2-D Tensor of shape `[height, width]`.
    seed: int. Used to create a random seed. See `tf.set_random_seed` for behavior.
    flip: int, flip & 1 > 0: left/right
               flip & 2 > 0: up/down
               flip & 4 > 0: front/back
    name: str
    extra: list of tuples, such as [(tensor, direction_axis_map)]
        tensor: tf.Tensor,
        direction_axis_map: dict, {direction: axis}, the direction can be chosen from
                            1 -> lr, 2 -> ud, 4 -> fb. With this dict, we can map the flips
                            of image and label to the specific axis of extra tensors.

    Returns
    -------
    image: tf.Tensor, with the same type and shape as parameter `image`.
    [when parameter `label` is not None] label: tf.Tensor
    [when parameter `extra` is not None] extra: a list of tf.Tensor

    Raises
    ------
    ValueError: if the shape of `image` not supported.
    """
    with tf.name_scope(name):
        if label is None:
            if flip & 1 > 0:
                image = tf.image.random_flip_left_right(image, seed=seed)
            if flip & 2 > 0:
                image = tf.image.random_flip_up_down(image, seed=seed)
            if flip & 4 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image = tf.reverse(image, axis=[0])
            return image
        elif extra is None:
            def lr_fn(img, lab):
                img = tf.image.flip_left_right(img)
                lab = tf.expand_dims(lab, axis=-1)
                lab = tf.image.flip_left_right(lab)
                lab = tf.squeeze(lab, axis=-1)
                return img, lab

            def ud_fn(img, lab):
                img = tf.image.flip_up_down(img)
                lab = tf.expand_dims(lab, axis=-1)
                lab = tf.image.flip_up_down(lab)
                lab = tf.squeeze(lab, axis=-1)
                return img, lab

            def fb_fn(img, lab):
                img = tf.reverse(img, axis=[0])
                lab = tf.reverse(lab, axis=[0])
                return img, lab

            def false_fn():
                return image, label

            if flip & 1 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label = lr_fn(image, label)
            if flip & 2 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label = ud_fn(image, label)
            if flip & 4 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label = fb_fn(image, label)
            return image, label
        else:
            def lr_fn2(img, lab, ext):
                img = tf.image.flip_left_right(img)
                lab = tf.expand_dims(lab, axis=-1)
                lab = tf.image.flip_left_right(lab)
                lab = tf.squeeze(lab, axis=-1)
                ext = [(tf.reverse(x, axis=[mp[1]]) if 1 in mp else x, mp) for x, mp in ext]
                return img, lab, ext

            def ud_fn2(img, lab, ext):
                img = tf.image.flip_up_down(img)
                lab = tf.expand_dims(lab, axis=-1)
                lab = tf.image.flip_up_down(lab)
                lab = tf.squeeze(lab, axis=-1)
                ext = [(tf.reverse(x, axis=[mp[2]]) if 2 in mp else x, mp) for x, mp in ext]
                return img, lab, ext

            def fb_fn2(img, lab, ext):
                img = tf.reverse(img, axis=[0])
                lab = tf.reverse(lab, axis=[0])
                ext = [(tf.reverse(x, axis=[mp[4]]) if 4 in mp else x, mp) for x, mp in ext]
                return img, lab, ext

            def false_fn2():
                return image, label, extra

            if flip & 1 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label, extra = lr_fn2(image, label, extra)
            if flip & 2 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label, extra = ud_fn2(image, label, extra)
            if flip & 4 > 0:
                if tf.random.uniform((), seed=seed) >= 0.5:
                    image, label, extra = fb_fn2(image, label, extra)
            extra = [x for x, _ in extra]
            return image, label, extra


def random_rotate(image, label, rotate_scale, seed=None):
    """
    Randomly rotate image(s) counterclockwise with given rotate_scale as stddev.
    Require package `tensorflow-addons`.

    Parameters
    ----------
    image: tf.Tensor, 4-D Tensor of shape `[batch_size, height, width, channels]`
                      3-D Tensor of shape `[height, width, channels]`
                      2-D Tensor of shape `[height, width]`
           Notice that only height-width plane is rotated and other axes are unchanged.
           Interpolation mode is 'bi-linear'.
    label: tf.Tensor, the same as image except interpolation mode 'nearest'.
    rotate_scale: float, a stddev used for sampling from normal distribution.
    seed: int, random seed.

    Returns
    -------
    image: tf.Tensor, rotated image
    label: tf.Tensor, rotated label
    """
    angle = tf.random.normal(shape=(), stddev=rotate_scale, seed=seed)
    angle = angle * (3.14159265 / 180)
    image = tfa.image.rotate(image, angle, interpolation="BILINEAR")
    label = tfa.image.rotate(label, angle, interpolation="NEAREST")
    return image, label


def augment_gamma(image, gamma_range, retain_stats=True, p_per_sample=1, epsilon=1e-7, seed=None):
    """
    Randomly augmentation with Gamma transformation.

    Parameters
    ----------
    image: tf.Tensor, any shape, perform on all the axes.
    gamma_range: tuple, range of gamma transformation. Such as (0.7, 1.5)
    retain_stats: bool, retain stats or not
    p_per_sample: float, [0, 1], probability for gamma in [gamma_range[0], 1)
                  1 - p_per_sample is for gamma in [1, gamma_range[1])
    epsilon: float, avoiding divide by zero
    seed: int, random seed

    Returns
    -------
    new_image: tf.Tensor, randomly augmented image
    """
    axes = list(range(len(image.shape)))
    if retain_stats:
        mn, variance = tf.nn.moments(image, axes=axes)
        sd = tf.math.sqrt(variance)

    rnd = tf.random.uniform((), 0, 1, dtype=tf.float32, seed=seed)
    if rnd < p_per_sample:
        gamma = tf.random.uniform((), gamma_range[0], 1, seed=seed)
    else:
        gamma = tf.random.uniform((), 1, gamma_range[1], seed=seed)
    minm = tf.reduce_min(image)
    rnge = tf.reduce_max(image) - minm
    new_image = tf.math.pow((image - minm) / (rnge + epsilon), gamma) * rnge + minm
    if retain_stats:
        new_mn, new_variance = tf.nn.moments(new_image, axes=axes)
        new_image = new_image - new_mn + mn
        new_image = new_image / (tf.math.sqrt(new_variance) + 1e-8) * sd
    return new_image


def gen_guide_2d(shape, center, stddev=None, euclidean=False):
    """
    Parameters
    ----------
    shape: tf.Tensor, two values
    center: tf.Tensor, Float tensor with shape [n, 2], 2 means (y, x)
    stddev: tf.Tensor, Float tensor with shape [n, 2], 2 means (y, x)
    euclidean: bool, use Euclidean distance or exponential edt

    Returns
    -------
    A batch of spatial guide image

    Warnings
    --------
    User must make sure stddev doesn't contain zero.

    Notes
    -----
    -1s in center and stddev are padding value and almost don't affect spatial guide
    """
    y = tf.range(shape[0])
    x = tf.range(shape[1])
    # Let n the number of tumors in current slice
    coords = tf.tile(tf.expand_dims(
        tf.stack(tf.meshgrid(y, x, indexing="ij"), axis=-1), axis=0),
        multiples=tf.concat((tf.shape(center)[:1], [1, 1, 1]), axis=0))     # [n, h, w, 2]
    coords = tf.cast(coords, tf.float32)
    center = tf.expand_dims(tf.expand_dims(center, axis=-2), axis=-2)       # [n, 1, 1, 2]
    if euclidean:
        d = tf.sqrt(tf.reduce_sum((coords - center) ** 2, axis=-1, keepdims=True))  # [n, h, w, 1]
        return tf.reduce_min(d, axis=0)     # [h, w, 1]

    # Add 1e-6 to avoid dividing by zero
    stddev = tf.expand_dims(tf.expand_dims(stddev, axis=-2), axis=-2) + 1e-6       # [n, 1, 1, 2]
    normalizer = 2. * stddev * stddev                                       # [n, 1, 1, 2]
    d = tf.exp(-tf.reduce_sum((coords - center) ** 2 / normalizer, axis=-1, keepdims=True))  # [n, h, w, 1]
    if tf.reduce_any(tf.math.is_nan(d)):
        raise ValueError(f"Nan encountered in `gen_guide_2d()`. stddev={stddev[:, 0, 0]}")
    return tf.reduce_max(d, axis=0)               # [h, w, 1]


def gen_guide_3d(shape, center, stddev=None, euclidean=False):
    """
    Parameters
    ----------
    shape: tf.Tensor, two values
    center: tf.Tensor, Float tensor with shape [n, 3], 3 means (z, y, x)
    stddev: tf.Tensor, Float tensor with shape [n, 3], 3 means (z, y, x)
    euclidean: bool, use Euclidean distance or exponential edt

    Returns
    -------
    A batch of spatial guide image

    Warnings
    --------
    User must make sure stddev doesn't contain zero.
    """
    z = tf.range(shape[0])
    y = tf.range(shape[1])
    x = tf.range(shape[2])
    # Let n the number of tumors in current slice
    coords = tf.tile(tf.expand_dims(
        tf.stack(tf.meshgrid(z, y, x, indexing="ij"), axis=-1), axis=0),
        multiples=tf.concat((tf.shape(center)[:1], [1, 1, 1, 1]), axis=0))  # [n, d, h, w, 3]
    coords = tf.cast(coords, tf.float32)
    center = tf.expand_dims(tf.expand_dims(tf.expand_dims(center, axis=-2), axis=-2), axis=-2)  # [n, 1, 1, 1, 3]
    if euclidean:
        d = tf.sqrt(tf.reduce_sum((coords - center) ** 2, axis=-1, keepdims=True))  # [n, d, h, w, 1]
        return tf.reduce_min(d, axis=0)     # [d, h, w, 1]

    # Add 1e-6 to avoid dividing by zero
    stddev = tf.expand_dims(tf.expand_dims(tf.expand_dims(stddev, axis=-2), axis=-2), axis=-2) + 1e-6  # [n, 1, 1, 1, 3]
    normalizer = 2. * stddev * stddev                                       # [n, 1, 1, 2]
    d = tf.exp(-tf.reduce_sum((coords - center) ** 2 / normalizer, axis=-1, keepdims=True))  # [n, d, h, w, 1]
    check_op = tf.Assert(tf.reduce_all(tf.logical_not(tf.math.is_nan(d))),
                         ["Nan encountered in `gen_guide_3d()`", stddev[:, 0, 0, 0]])
    with tf.control_dependencies([check_op]):
        gd = tf.reduce_max(d, axis=0)               # [d, h, w, 1]
    return gd
