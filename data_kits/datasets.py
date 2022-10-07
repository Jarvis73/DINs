import math

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tensorflow as tf
import tqdm
from scipy.ndimage.morphology import binary_erosion

from data_kits import nf_kits
from data_kits import np_ops, tf_ops

_data_cache = None


def check_size(shape, *args):
    if isinstance(shape, (list, tuple)):
        res = [None if x <= 0 else x for x in shape]
        if isinstance(shape, tuple):
            res = tuple(res)
        return res
    elif isinstance(shape, (int, float)):
        return tuple([None if shape <= 0 else shape] + [None if x <= 0 else x for x in args])
    else:
        raise TypeError(f"Unsupported type of shape: {type(shape)}")


def loads(opt, logger, set_key):
    data_list = nf_kits.load_split(set_key, opt.fold, opt.fold_path)

    if set_key in ["train", "eval_online"]:
        return get_loader_train(opt, logger, data_list, set_key)
    elif set_key == "eval":
        return get_loader_eval(opt, logger, data_list)
    elif set_key == "test":
        if opt.use_box:
            return get_loader_test_with_box(opt, logger, data_list)
        return get_loader_test(opt, logger, data_list)
    else:
        raise ValueError


def load_data(logger=None):
    global _data_cache

    if _data_cache is None:
        _data_cache = nf_kits.load_data(logger)
    return _data_cache


def prob_by_area(x):
    if x < 50:
        return 0.04
    if x < 300:
        return 0.2
    return 1


def del_circle(image, ctr, radius):
    """
    Delete a circle from an image. Set the values in the circle to zero.
    Notice that this function manipulate the input `image`.

    Parameters
    ----------
    image: np.ndarray
        [depth, height, width]
    ctr: np.ndarray
        length 3, circle center
    radius: int
        circle radius
    """
    _, height, width = image.shape
    y1 = max(ctr[-2] - radius, 0)
    y2 = min(ctr[-2] + radius + 1, height)
    x1 = max(ctr[-1] - radius, 0)
    x2 = min(ctr[-1] + radius + 1, width)
    rcy, rcx, rh, rw = ctr[-2] - y1, ctr[-1] - x1, y2 - y1, x2 - x1   # relative center y, x, relative height, width
    y, x = np.meshgrid(np.arange(rh), np.arange(rw), indexing="ij", sparse=True)
    circle = (x - rcx) ** 2 + (y - rcy) ** 2 > radius ** 2
    image[ctr[0], y1:y2, x1:x2] *= circle   # Only operate on single slice


def inter_simulation(mask, sampler, margin=3, step=10, n=11, bg=False, d=40, strategy=0):
    """
    Interaction simulation, including positive points and negative points

    Parameters
    ----------
    mask: np.ndarray
        binary mask, foreground points sampled from label=1 and bg from label=0
    sampler: np.random.RandomState
    margin: int
        margin band width in which no points are sampled
    step: int
        minimal distance between multiple interactions
    n: int
        maximum number of interactions
    bg: bool
        True for border_value=1, False for border_value=0 in binary erosion
    d: int
        band width outside the object
    strategy: int
        value in [0, 1, 2],
        0: random in whole fg
        1: random in band
        2: surround the object evenly in band

    Returns
    -------
    fg_pts: np.ndarray, shape [m, 3], ret_type, coordinates of the positive points
    bg_pts: np.ndarray, shape [n, 3], ret_type, coordinates of the negative points
    """
    small = False
    first = True
    all_pts = []
    struct = np.zeros((3, 3, 3), dtype=np.bool)
    struct[1] = True
    g = binary_erosion(mask, struct, iterations=margin, border_value=bg)
    if bg and strategy != 0:
        # let g be the background band
        g = g ^ binary_erosion(g, struct, iterations=d, border_value=bg)
    if not g.max():
        # tumor is too small, discard `margin`
        g = mask.copy()
        small = True

    # determine sample number
    inter_num = sampler.randint(int(not bg), n)
    for _ in range(inter_num):
        ctr = np.stack(np.where(g), axis=1)
        if not small:
            if first or strategy in [0, 1]:
                i = sampler.choice(ctr.shape[0])
            else:  # strategy == 2
                dist = ctr.reshape(-1, 1, 3) - np.asarray(all_pts).reshape(1, -1, 3)
                # choose the farthest point
                i = np.argmax(np.sum(dist ** 2, axis=-1).min(axis=1))
            ctr = ctr[i]   # center z, y, x
        else:
            # For small object, directly use the center
            ctr = ctr.mean(axis=0).round().astype(np.int32)
        first = False
        all_pts.append(ctr)
        del_circle(g, ctr, step)
        if small or g.max() == 0:  # Should not add more points
            break

    return np.asarray(all_pts, dtype=np.float32).reshape(-1, 3)


def get_pts(lab_patch, num_inters_max, sampler):
    """
    get interactive points

    Parameters
    ----------
    lab_patch: np.ndarray
        label patch, np.uint8
    num_inters_max: int
        maximum number of interactions
    sampler: np.random.RandomState

    Returns
    -------
    fg_pts: np.ndarray, with shape [None, 3], foreground points
    bg_pts: np.ndarray, with shape [None, 3], background points
    """
    if lab_patch.max() > 0:
        fg_pts = inter_simulation(lab_patch, sampler, n=num_inters_max, bg=False, strategy=0)
    else:
        fg_pts = np.zeros((0, 3), dtype=np.float32)

    strategy = 1 if sampler.uniform() > 0.5 else 2
    bg_pts = inter_simulation(1 - lab_patch, sampler, n=num_inters_max, bg=True, strategy=strategy)

    return fg_pts, bg_pts


def data_processing(img, lab, *pts, opt=None, logger=None, mode=None):
    """
    Pre-process training data with tensorflow API

    Parameters
    ----------
    img: tf.Tensor
        with shape [None, None, None, channel]
    lab: tf.Tensor
        with shape [None, None, None]
    pts: tuple
        containing two tf.Tensor, with shape [None, 3]
    opt:
    logger:
    mode: str
        must in [train|val]

    Returns
    -------
    img: tf.Tensor if guide_type == "none", else a tuple with two tf.Tensor (img, guide)
        img: with shape [depth, height, width, channel]
        guide: with shape [depth, height, width, 2]
    lab: tf.Tensor, with shape [depth, height, width]
    """
    # z_score
    img = tf_ops.z_score(img)

    target_shape = tf.convert_to_tensor([opt.depth, opt.height, opt.width], dtype=tf.int32)
    # Only resize height and width. Keep depth unchanged.
    img = tf.image.resize(img, (opt.height, opt.width))

    def pts_to_guide(ctr, std):
        if tf.shape(ctr)[0] > 0:
            if opt.guide == "exp":
                stddev = tf.maximum(tf.ones(tf.shape(ctr), tf.float32) * std, 0.1)
                gd = tf_ops.gen_guide_3d(target_shape, ctr, stddev, euclidean=False)
            elif opt.guide == "euc":
                gd = tf_ops.gen_guide_3d(target_shape, ctr, euclidean=True)
            elif opt.guide == "geo":
                int_ctr = tf.cast(ctr, tf.int32)
                # Convert numpy operations to tensorflow operations, sacrifice some performance
                gd = tf.py_function(
                    np_ops.gen_guide_geo_3d, [img[..., 0], int_ctr, opt.geo_lamb, opt.geo_iter],
                    tf.float32, name="GeoDistLarge")
                gd.set_shape((opt.depth, opt.height, opt.width))
                gd = tf.expand_dims(gd, axis=-1)
            else:
                raise ValueError(f"Unsupported guide type: {opt.guide}")
            guide = tf.cast(gd, tf.float32)
        else:
            guide = tf.zeros(tf.concat([tf.shape(img)[:-1], [1]], axis=0), tf.float32)
        return guide

    fg_pts, bg_pts = pts
    scale = tf.cast(target_shape, tf.float32) / tf.cast(tf.shape(lab), tf.float32)
    fg_pts = fg_pts * scale
    bg_pts = bg_pts * scale
    fg_guide = pts_to_guide(fg_pts, opt.exp_stddev)
    bg_guide = pts_to_guide(bg_pts, opt.exp_stddev)
    logger.info(f"Use guide with {opt.guide} distance" +
                f", stddev={tuple(opt.exp_stddev)}" * (opt.guide == "exp"))
    img = tf.concat([img, fg_guide, bg_guide], axis=-1)

    if mode == "train":
        if opt.flip > 0:
            img, lab = tf_ops.random_flip(img, lab, flip=opt.flip)
        if opt.rotate > 0:
            # Only rotate height and width. Keep depth unchanged.
            lab = tf.expand_dims(lab, axis=-1)
            img, lab = tf_ops.random_rotate(img, lab, rotate_scale=opt.rotate)
            lab = tf.squeeze(lab, axis=-1)

    img, sp_guide = tf.split(img, [opt.channel, 2], axis=-1)

    lab = tf.expand_dims(lab, axis=-1)
    lab = tf.image.resize(lab, (opt.height, opt.width), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.squeeze(lab, axis=-1)

    if mode == "train":
        img = tf_ops.augment_gamma(img, gamma_range=(0.7, 1.5), retain_stats=True, p_per_sample=0.3)

    img = (img, sp_guide)
    return img, lab


def volume_crop(volume, center, shape, extend_z=(0, 0)):
    depth, height, width = volume.shape
    half_d, half_h, half_w = shape[0] // 2, shape[1] // 2, shape[2] // 2
    z1 = min(max(center[0] - half_d, 0), depth - shape[0])
    z2 = z1 + shape[0]
    y1 = min(max(center[1] - half_h, 0), height - shape[1])
    y2 = y1 + shape[1]
    x1 = min(max(center[2] - half_w, 0), width - shape[2])
    x2 = x1 + shape[2]
    slices = (slice(z1, z2), slice(y1, y2), slice(x1, x2))

    pad_z1 = max(0, extend_z[0] - z1)
    pad_z2 = max(0, extend_z[1] + z2 - depth)
    z1 = max(0, z1 - extend_z[0])
    z2 = min(z2 + extend_z[1], depth)
    img = volume[z1:z2, y1:y2, x1:x2]
    img = np.pad(img, ((pad_z1, pad_z2), (0, 0), (0, 0)))

    return img, slices


def gen_batch(opt, data_list, mode, sampler):
    """ Batch sampler

    Parameters
    ----------
    opt:
        Configurations
    data_list: pd.DataFrame
        with column [split, pid, remove]
    mode: str
        train or val
    sampler: np.random.RandomState
        random state

    Returns
    -------
    A generator:
        img_patch: np.ndarray
            with shape [None, None, None, 1], tf.float32
        lab_patch: np.ndarray
            with shape [None, None, None], tf.int32
        fg_pts: np.ndarray
            with shape [None, 3], np.float32
        bg_pts: np.ndarray
            with shape [None, 3], np.float32
    """
    train = mode == "train"
    data = load_data()
    data_list = data_list[[True if pid in data else False for pid in data_list.pid]]
    # dataset containing nf (remove benign scans)
    data_list['nf'] = [True if len(data[pid]['lab_rng']) > 1 else False for pid in data_list.pid]
    nf_set = data_list[data_list.nf]

    # Set minimum sample percentage containing tumors
    force_tumor = math.ceil(opt.bs * opt.tumor_percent)

    target_size = np.array([opt.depth, opt.height, opt.width], dtype=np.float32)
    if train:
        zoom = opt.zoom
    else:
        zoom = ((opt.zoom[0] + opt.zoom[1]) / 2, ) * 2

    while True:
        nf = nf_set.sample(
            n=force_tumor, replace=False, weights=None, random_state=sampler)
        nf['flag'] = [1] * len(nf.index)
        rem = data_list[~data_list.index.isin(nf.index)].sample(
            n=opt.bs - force_tumor, replace=False, weights=None, random_state=sampler)
        rem['flag'] = [0] * len(rem.index)
        batch = pd.concat([nf, rem])

        for i, sample in batch.iterrows():     # columns [split, pid, nf(bool)]
            crop_shape = (target_size * (1, *sampler.uniform(*zoom, 2))).astype(np.int32)
            d, h, w = data[sample.pid]['img'].shape  # volume shape
            if sample.flag == 1:
                # choose a foreground pixel
                i = sampler.choice(data[sample.pid]['pos'].shape[0])
                pz, py, px = data[sample.pid]['pos'][i]
            else:
                # choose a random pixel
                pz = sampler.randint(d)
                py = sampler.randint(h)
                px = sampler.randint(w)
            img_patch, slices = volume_crop(data[sample.pid]['img'], (pz, py, px), crop_shape)
            lab_patch = np.clip(data[sample.pid]['lab'][slices], 0, 1)  # np.uint8

            img_patch = img_patch[..., None]
            yield_list = (img_patch.astype(np.float32), lab_patch.astype(np.int32))
            yield_list = yield_list + get_pts(lab_patch, opt.num_inters_max, sampler)

            yield yield_list


def get_loader_train(opt, logger, data_list, set_key):
    load_data(logger)     # load data to cache
    bs = opt.bs

    def train_gen():
        infinity_generator = gen_batch(opt, data_list, "train", np.random.RandomState())
        ranges = tqdm.tqdm(
            range(opt.train_n), desc="Train", unit_scale=1. / bs,
            dynamic_ncols=True)
        for _ in ranges:
            yield next(infinity_generator)

    def val_gen():
        infinity_generator = gen_batch(opt, data_list, "val", sampler=np.random.RandomState(1234))
        ranges = tqdm.tqdm(
            range(opt.val_n), desc="Val", unit_scale=1. / bs,
            dynamic_ncols=True)
        for _ in ranges:
            yield next(infinity_generator)

    data_gen = train_gen if set_key == "train" else val_gen

    input_shape = [
        check_size(bs, opt.depth, opt.height, opt.width, opt.channel),
        check_size(bs, opt.depth, opt.height, opt.width, 2),
    ]
    output_types = (
        tf.float32,
        tf.int32,
        tf.float32,
        tf.float32
    )
    output_shapes = (
        tf.TensorShape([None, None, None, opt.channel]),
        tf.TensorShape([None, None, None]),
        tf.TensorShape([None, 3]),
        tf.TensorShape([None, 3]),
    )

    def map_fn(*args):
        func = data_processing(*args, opt=opt, logger=logger, mode=set_key)
        return func

    dataset = tf.data.Dataset.from_generator(data_gen, output_types, output_shapes)\
        .map(map_fn, num_parallel_calls=bs * 2)\
        .batch(bs, drop_remainder=True)\
        .prefetch(2)

    logger.info(f"           ==> Dataloader 3D for {set_key}")
    return dataset, input_shape


def eval_gen(opt, data_fn, data_list):
    for i, (_, sample) in enumerate(data_list.iterrows(), start=1):
        if 0 <= opt.test_n < i:
            break
        volume, label, meta = data_fn(sample.pid)
        if label.max() == 0:
            continue
        assert volume.shape == label.shape, f"{volume.shape} vs {label.shape}"
        zoom_scale = np.array(
            (1, opt.test_height / volume.shape[1], opt.test_width / volume.shape[2]), np.float32)
        resized_volume = ndi.zoom(volume, zoom_scale, order=1)
        if resized_volume.shape[0] % 2 != 0:
            resized_volume = np.pad(
                resized_volume, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        normed_volume = np_ops.z_score(resized_volume)
        yield sample, meta, normed_volume, label


def get_loader_eval(opt, logger, data_list):
    data = load_data(logger)
    data_list = data_list[[True if item.pid in data and item.remove != 1 else False
                          for _, item in data_list.iterrows()]].copy()
    data_list['nf'] = [True if len(data[pid]['lab_rng']) > 1 else False for pid in data_list.pid]
    data_list = data_list[data_list.nf]
    data = nf_kits.slim_labels(data, logger)

    def data_fn(pid):
        return data[pid]["img"].astype(np.float32), data[pid]["slim"], data[pid]["meta"]

    generator = eval_gen(opt, data_fn, data_list)
    input_shape = [
        check_size(1, opt.test_depth, opt.test_height, opt.test_width, opt.channel),
        check_size(1, opt.test_depth, opt.test_height, opt.test_width, 2),
    ]

    return generator, input_shape


def get_loader_test(opt, logger, data_list):
    data = nf_kits.load_test_data_paths()    # Only contain data paths

    data_list = data_list[[True if item.pid in data and item.remove != 1 else False
                           for _, item in data_list.iterrows()]].copy()

    def data_fn(pid):
        volume = nf_kits.read_nii(data[pid]["img_path"])[1].astype(np.float32)
        meta, label = nf_kits.read_nii(data[pid]["lab_path"], np.int8)
        label = np.clip(label, 0, 1)
        if volume.min() < 0:
            volume[volume < 0] = 0
        return volume, label, meta

    generator = eval_gen(opt, data_fn, data_list)
    input_shape = [
        check_size(1, opt.test_depth, opt.test_height, opt.test_width, opt.channel),
        check_size(1, opt.test_depth, opt.test_height, opt.test_width, 2),
    ]

    return generator, input_shape


def eval_gen_with_box(opt, data_fn, data_list, box_ds):
    for _, sample in data_list.iterrows():
        volume, label, meta = data_fn(sample.pid)
        if label.max() == 0:
            continue
        assert volume.shape == label.shape, f"{volume.shape} vs {label.shape}"
        depth, height, width = volume.shape
        # Inference with each of the boxes
        case_box_ds = box_ds[box_ds['pid'] == sample.pid]
        total = len(case_box_ds)
        for bid, (_, box) in enumerate(case_box_ds.iterrows()):
            if box['z1'] < 0:
                # Use the full image
                img_patch = volume
                lab_patch = label
                crop_box = None
                mask = None
                t_height = opt.test_height
                t_width = opt.test_width
                max_iter = 20
            else:
                pid, z1, z2, y1, y2, x1, x2, max_iter = box
                if z2 - z1 < 6:
                    half = (6 - z2 + z1) // 2
                    z1 = min(max(z1 - half, 0), depth - 6)
                    z2 = z1 + 6
                if y2 - y1 < 128:
                    half = (128 - y2 + y1) // 2
                    y1 = min(max(y1 - half, 0), height - 128)
                    y2 = y1 + 128
                if x2 - x1 < 128:
                    half = (128 - x2 + x1) // 2
                    x1 = min(max(x1 - half, 0), width - 128)
                    x2 = x1 + 128
                img_patch = volume[z1:z2, y1:y2, x1:x2]
                lab_patch = label[z1:z2, y1:y2, x1:x2]
                mask = np.zeros_like(label, np.int32)
                _, a, b, c, d, e, f, _ = box
                mask[a:b, c:d, e:f] = 1
                mask = mask[z1:z2, y1:y2, x1:x2]
                lab_patch *= mask
                crop_box = [bid, total, (slice(z1, z2), slice(y1, y2), slice(x1, x2))]
                t_height = int(round(img_patch.shape[1] / 16) * 16)
                t_width = int(round(img_patch.shape[2] / 16) * 16)

            zoom_scale = np.array(
                (1, t_height / img_patch.shape[1], t_width / img_patch.shape[2]), np.float32)
            resized_patch = ndi.zoom(img_patch, zoom_scale, order=1)
            if mask is not None:
                mask = ndi.zoom(mask, zoom_scale, order=0)
            if resized_patch.shape[0] % 2 != 0:
                resized_patch = np.pad(
                    resized_patch, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
            normed_patch = np_ops.z_score(resized_patch)
            yield sample, meta, normed_patch, (lab_patch, label), crop_box, mask, max_iter


def get_loader_test_with_box(opt, logger, data_list):
    _ = logger
    data = nf_kits.load_test_data_paths()   # Only contain data paths
    box_ds = pd.read_csv(opt.bbox_path)
    box_pids = list(box_ds["pid"])
    data_list = data_list[[True if item.pid in data and item.pid in box_pids and item.remove != 1
                           else False for _, item in data_list.iterrows()]].copy()

    def data_fn(pid):
        volume = nf_kits.read_nii(data[pid]["img_path"])[1].astype(np.float32)
        meta, label = nf_kits.read_nii(data[pid]["lab_path"], np.int8)
        label = np.clip(label, 0, 1)
        return volume, label, meta

    generator = eval_gen_with_box(opt, data_fn, data_list, box_ds)
    input_shape = [(1, None, None, None, 1), (1, None, None, None, 2)]
    return generator, input_shape
