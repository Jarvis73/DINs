# Copyright 2019-2020 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import pickle
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tqdm

ROOT = Path(__file__).parents[1]
DATA_ROOT = ROOT / "data/NF"


def read_nii(file_name, out_dtype=np.int16, special=False, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh
    affine = vh.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])
    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, special=False, affine=None):
    if header is not None:
        affine = header.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    trans_bk = [np.argwhere(np.array(trans[::-1]) == i)[0][0] for i in range(3)]

    if special:
        data = np.flip(data, axis=2)
    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, trans_bk).astype(out_dtype)
    if header is None and affine is not None:
        out = nib.Nifti1Image(out_image, affine=affine)
    else:
        out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))


def load_data(logger):
    data_dir = DATA_ROOT / "nii_NF"
    path_list = list(data_dir.glob("volume*"))

    logger.info(' ' * 11 + f"==> Loading data ({len(path_list)} examples) ...")
    cache_path = DATA_ROOT / "cache.pkl.gz"
    if cache_path.exists():
        logger.info(' ' * 11 + f"==> Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            _data_cache = pickle.loads(data)
        logger.info(' ' * 11 + "==> Finished!")
        return _data_cache

    _data_cache = {}
    for path in tqdm.tqdm(path_list):
        pid = path.name.split(".")[0].split("-")[-1]
        header, volume = read_nii(path)
        la_path = path.parent / path.name.replace("volume", "segmentation")
        _, label = read_nii(la_path)
        assert volume.shape == label.shape, f"{volume.shape} vs {label.shape}"
        _data_cache[int(pid)] = {"im_path": path.absolute(),
                                 "la_path": la_path.absolute(),
                                 "img": volume,
                                 "lab": label.astype(np.uint8),
                                 "pos": np.stack(np.where(label > 0), axis=1),
                                 "meta": header,
                                 "lab_rng": np.unique(label)}
    with cache_path.open("wb") as f:
        logger.info(' ' * 11 + f"==> Saving data cache to {cache_path}")
        cache_s = pickle.dumps(_data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    logger.info(' ' * 11 + "==> Finished!")
    return _data_cache


def load_split(set_key, test_fold, fold_path):
    if set_key in ["train", "eval_online", "eval"]:
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        val_split = folds.loc[folds.split == test_fold]
        if set_key != "train":
            return val_split
        train_folds = list(range(5))
        train_folds.remove(test_fold)
        train_split = folds.loc[folds.split.isin(train_folds)]
        return train_split
    elif set_key == "test":
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        test_split = folds.loc[folds.split == 0]
        return test_split
    else:
        raise ValueError(f"`set_key` supports [train|eval_online|eval|test|extra], got {set_key}")


def filter_tiny_nf(mask):
    struct2 = ndi.generate_binary_structure(2, 1)
    for i in range(mask.shape[0]):
        res, n_obj = ndi.label(mask[i], struct2)
        size = np.bincount(res.flat)
        for j in np.where(size <= 2)[0]:
            mask[i][res == j] = 0

    struct3 = ndi.generate_binary_structure(3, 2)
    res, n_obj = ndi.label(mask, struct3)
    size = np.bincount(res.flat)
    for i in np.where(size <= 5)[0]:
        mask[res == i] = 0
    return mask


def slim_labels(data, logger):
    slim_labels_path = DATA_ROOT / "slim_labels.pkl.gz"
    if slim_labels_path.exists():
        logger.info(' ' * 11 + f"==> Loading slimmed label cache from {slim_labels_path}")
        with slim_labels_path.open("rb") as f:
            new_labels = pickle.loads(zlib.decompress(f.read()))
        for i in data:
            data[i]['slim'] = new_labels[i]
        logger.info(' ' * 11 + "==> Finished!")
    else:
        new_labels = {}
        logger.info(' ' * 11 + f"==> Saving slimmed label cache to {slim_labels_path}")
        for i, item in data.items():
            new_labels[i] = filter_tiny_nf(np.clip(item['lab'], 0, 1).copy())
            data[i]['slim'] = new_labels[i]
        with slim_labels_path.open("wb") as f:
            f.write(zlib.compress(pickle.dumps(new_labels, pickle.HIGHEST_PROTOCOL)))
        logger.info(' ' * 11 + "==> Finished!")

    return data


def load_test_data_paths():
    data_dir = DATA_ROOT / "test_NF"
    path_list = list(data_dir.glob("*img.nii.gz"))
    dataset = {}
    for path in path_list:
        pid = int(path.name.split("-")[0])
        dataset[pid] = {"img_path": path, "lab_path": path.parent / path.name.replace("img", "mask")}
    return dataset
