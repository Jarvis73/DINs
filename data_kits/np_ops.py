"""
Some utility functions using numpy
"""

import numpy as np


def z_score(img):
    """
    Z-score image normalization.

    Parameters
    ----------
    img: np.ndarray, it should be float format. Normalization is performed on all the axes

    Returns
    -------
    new_img: Normalized image
    """
    msk = img > 0
    tmp = img[msk]
    new_img = img.copy()
    new_img[msk] = (tmp - tmp.mean()) / (tmp.std() + 1e-8)
    return new_img


def _all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def one_hot(indices, depth, axis):
    shape = indices.shape[:axis] + (depth,) + indices.shape[axis:]
    out = np.zeros(shape, dtype=int)
    out[_all_idx(indices, axis=axis)] = 1
    return out


def gen_guide_nd(shape, centers, stddevs=None, indexing="ij", keepdims=False,
                 euclidean=False):
    """
    Parameters
    ----------
    shape: list, two values
    centers: ndarray, Float ndarray with shape [n, d], d means (x, y, ...)
    stddevs: ndarray, Float ndarray with shape [n, d], d means (x, y, ...)
    indexing: {'xy', 'ij'}, optional, Cartesian ('xy') or matrix ('ij', default) indexing of output.
    keepdims: bool, Keep final dimension
    euclidean: bool, return euclidean distance or exponential distance

    Returns
    -------
    A batch of spatial guide image with shape [h, w, 1] for 2D and [d, h, w, 1] for 3D

    """
    if len(centers) == 0:
        res = np.zeros(shape, np.float32)
        if keepdims:
            res = res[..., None]
        return res
    centers = np.asarray(centers, np.float32)                                   # [n, 3]
    assert centers.ndim == 2, centers.shape
    coords = [np.arange(0, s) for s in shape]
    coords = np.tile(
        np.stack(np.meshgrid(*coords, indexing=indexing), axis=-1)[None],
        reps=[centers.shape[0]] + [1] * (centers.shape[1] + 1))                 # [n, d, h, w, 3]
    coords = coords.astype(np.float32)
    c_sh = centers.shape
    centers = centers.reshape(c_sh[:1] + (1,) * c_sh[-1] + c_sh[-1:])           # [n, 1, 1, 1, 3]
    if euclidean:
        d = np.sqrt(np.sum((coords - centers) ** 2, axis=-1, keepdims=keepdims))
        return np.min(d, axis=0)

    stddevs = np.asarray(stddevs, np.float32)                                   # [n, 3]
    stddevs = stddevs.reshape(c_sh[:1] + (1,) * c_sh[-1] + c_sh[-1:])           # [n, 1, 1, 1, 3]
    normalizer = 2 * stddevs * stddevs                                          # [n, 1, 1, 1, 3]
    d = np.exp(-np.sum((coords - centers) ** 2 / normalizer, axis=-1, keepdims=keepdims))   # [n, d, h, w, 1]
    return np.max(d, axis=0)                                                    # [d, h, w, 1]


def gen_guide_geo_nd(image, centers, lamb, iter_=1):
    if len(centers) == 0:
        return np.zeros_like(image, np.float32)

    import GeodisTK

    S = np.zeros_like(image, np.uint8)
    if image.ndim == 2:
        S[centers[:, 0], centers[:, 1]] = 1
        return GeodisTK.geodesic2d_raster_scan(image, S, lamb, iter_)
    elif image.ndim == 3:
        S[centers[:, 0], centers[:, 1], centers[:, 2]] = 1
        spacing = [1., 1., 1.]
        return GeodisTK.geodesic3d_raster_scan(image, S, spacing, lamb, iter_)
    else:
        raise ValueError


gen_guide_geo_3d = gen_guide_geo_nd
gen_guide_geo_2d = gen_guide_geo_nd


def compute_robust_moments(binary_image, isotropic=False, indexing="ij", min_std=0.):
    """
    Compute robust center and standard deviation of a binary image(0: background, 1: foreground).

    Support n-dimension array.

    Parameters
    ----------
    binary_image: ndarray
        Input an image
    isotropic: boolean
        Compute isotropic standard deviation or not.
    indexing: {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.
        See Notes for more details.
    min_std: float
        Set stddev lower bound

    Returns
    -------
    center: ndarray
        A vector with dimension = `binary_image.ndim`. Median of all the points assigned 1.
    std_dev: ndarray
        A vector with dimension = `binary_image.ndim`. Standard deviation of all the points assigned 1.

    Notes
    -----
    All the points assigned 1 are considered as a single object.

    """
    ndim = binary_image.ndim
    coords = np.nonzero(binary_image)
    points = np.asarray(coords).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0] * ndim, dtype=np.float32), np.array([-1.0] * ndim, dtype=np.float32)
    points = np.transpose(points)       # [pts, 2], 2: (i, j)
    center = np.median(points, axis=0)  # [2]

    # Compute median absolute deviation(short for mad) to estimate standard deviation
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad] * ndim)
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826 * mad
    std_dev = np.maximum(std_dev, [min_std] * ndim)
    if not indexing or indexing == "xy":
        return center[::-1], std_dev[::-1]
    elif indexing == "ij":
        return center, std_dev
    else:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
