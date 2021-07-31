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

ROOT = Path(__file__).parent.parent.parent
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

    logger.info(f"Loading data ({len(path_list)} examples) ...")
    cache_path = DATA_ROOT / "cache.pkl.gz"
    if cache_path.exists():
        logger.info(f"Loading data cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            _data_cache = pickle.loads(data)
        logger.info("Finished!")
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
        logger.info(f"Saving data cache to {cache_path}")
        cache_s = pickle.dumps(_data_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    logger.info("Finished!")
    return _data_cache


def pre_filter_data(data, filter_thresh, connectivity=3, down_sampling=False):
    """ For object-based segmentation tasks.
    Pre-compute connected components and remove small objects
    """
    _pre_filter_cache = None

    cache_path = DATA_ROOT / ("pre-filter.pkl.gz" if not down_sampling else "pre-filter_ds.pkl.gz")
    if cache_path.exists():
        logger.info(f"Loading pre-filter cache from {cache_path}")
        with cache_path.open("rb") as f:
            data = zlib.decompress(f.read())
            _pre_filter_cache = pickle.loads(data)
        logger.info("Finished!")
        return _pre_filter_cache

    _pre_filter_cache = {}
    for pid in data:
        mask = data[pid]["lab"]
        struct = ndi.generate_binary_structure(3, connectivity)
        labeled, n_obj = ndi.label(mask, struct)
        slices = ndi.find_objects(labeled)
        obj_list = []
        for i, sli in enumerate(slices):
            patch = labeled[sli]
            z, y, x = np.where(patch == i + 1)
            if z.shape[0] < filter_thresh:
                patch[z, y, x] = 0
            else:
                obj_list.append(np.stack((z, y, x), axis=1))
        better_label = np.clip(labeled, 0, 1)
        _pre_filter_cache[pid] = {"lab": better_label,
                                  "obj_list": obj_list}
    with cache_path.open("wb") as f:
        logger.info(f"Saving pre-filter cache to {cache_path}")
        cache_s = pickle.dumps(_pre_filter_cache, pickle.HIGHEST_PROTOCOL)
        f.write(zlib.compress(cache_s))
    logger.info("Finished!")
    return _pre_filter_cache


def load_split(set_key, test_fold):
    if set_key in ["train", "val", "eval"]:
        fold_path = DATA_ROOT / "split.csv"
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        val_split = folds.loc[folds.split == test_fold]
        if set_key != "train":
            return val_split
        train_folds = list(range(5))
        train_folds.remove(test_fold)
        train_split = folds.loc[folds.split.isin(train_folds)]
        return train_split
    elif set_key == "test":
        fold_path = DATA_ROOT / "split_test.csv"
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        test_split = folds.loc[folds.split == 0]
        return test_split
    elif set_key == "extra":    # The dataset with 45 cases of 15 patients
        fold_path = DATA_ROOT / "split_extra.csv"
        folds = pd.read_csv(str(fold_path)).fillna(0).astype(int)
        test_split = folds.loc[folds.split == 0]
        return test_split
    else:
        raise ValueError(f"`set_key` supports [train|val|test|extra], got {set_key}")


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
        logger.info(f"Loading slimmed label cache from {slim_labels_path}")
        with slim_labels_path.open("rb") as f:
            new_labels = pickle.loads(zlib.decompress(f.read()))
        for i in data:
            data[i]['slim'] = new_labels[i]
        logger.info("Finished!")
    else:
        new_labels = {}
        logger.info(f"Saving slimmed label cache to {slim_labels_path}")
        for i, item in data.items():
            new_labels[i] = filter_tiny_nf(np.clip(item['lab'], 0, 1).copy())
            data[i]['slim'] = new_labels[i]
        with slim_labels_path.open("wb") as f:
            f.write(zlib.compress(pickle.dumps(new_labels, pickle.HIGHEST_PROTOCOL)))
        logger.info("Finished!")

    return data


def load_test_data_paths():
    data_dir = DATA_ROOT / "test_NF"
    path_list = list(data_dir.glob("*img.nii.gz"))
    dataset = {}
    for path in path_list:
        pid = int(path.name.split("-")[0])
        dataset[pid] = {"img_path": path, "lab_path": path.parent / path.name.replace("img", "mask")}
    return dataset


extra_name_mapping = {
    "---Abdomen1__20080620-img.nii.gz": 0,
    "---Abdomen1__20101129-img.nii.gz": 1,
    "---Abdomen1__20130625-img.nii.gz": 2,
    "---Airway1__20031216-img.nii.gz": 3,
    "---Airway1__20041020-img.nii.gz": 4,
    "---Airway1__20060907-img.nii.gz": 5,
    "---Airway2__20080707-img.nii.gz": 6,
    "---Airway2__20110124-img.nii.gz": 7,
    "---Airway2__20130204-img.nii.gz": 8,
    "---Back1__20070330-img.nii.gz": 9,
    "---Back1__20081117-img.nii.gz": 10,
    "---Back1__20100323-img.nii.gz": 11,
    "---Brachial-plexus1__20130205-img.nii.gz": 12,
    "---Br-plexus1__20120223-img.nii.gz": 13,
    "---Br-plexus1__20120625-img.nii.gz": 14,
    "---Chest2__20011227-img.nii.gz": 15,
    "---Chest2__20050914-img.nii.gz": 16,
    "---Chest2__20080918-img.nii.gz": 17,
    "---Chest3__20081222-img.nii.gz": 18,
    "---Chest3__20110602-img.nii.gz": 19,
    "---Chest3__20131122-img.nii.gz": 20,
    "---Face1__20100719-img.nii.gz": 21,
    "---Face1__20110418-img.nii.gz": 22,
    "---Face1__20120924-img.nii.gz": 23,
    "---Leg1__20080714-img.nii.gz": 24,
    "---Leg1__20100726-img.nii.gz": 25,
    "---Leg1__20110228-img.nii.gz": 26,
    "---Neck1__20020726-img.nii.gz": 27,
    "---Neck1__20040315-img.nii.gz": 28,
    "---Neck1__20050527-img.nii.gz": 29,
    "---Orbit1__20030225-img.nii.gz": 30,
    "---Orbit1__20050217-img.nii.gz": 31,
    "---Orbit1__20061016-img.nii.gz": 32,
    "---Orbit2__20090403-img.nii.gz": 33,
    "---Orbit2__20121018-img.nii.gz": 34,
    "---Orbit2__20140520-img.nii.gz": 35,
    "---Pelvis1__20030916-img.nii.gz": 36,
    "---Pelvis1__20060109-img.nii.gz": 37,
    "---Pelvis1__20100726-img.nii.gz": 38,
    "---Pelvis2__20090114-img.nii.gz": 39,
    "---Pelvis2__20100112-img.nii.gz": 40,
    "---Pelvis2__20120423-img.nii.gz": 41,
    "---Thigh1__20071019-img.nii.gz": 42,
    "---Thigh1__20100712-img.nii.gz": 43,
    "---Thigh1__20120106-img.nii.gz": 44,
}


def load_extra_data_paths():
    data_dir = DATA_ROOT / "NCI_NF1_InaLabeled"
    path_list = list(data_dir.glob("*img.nii.gz"))
    dataset = {}
    for path in path_list:
        pid = extra_name_mapping[path.name]
        dataset[pid] = {"img_path": path, "lab_path": path.parent / path.name.replace("img", "mask")}
    return dataset


def load_box_csv():
    box_file = DATA_ROOT / "nf_box.csv"
    box_df = pd.read_csv(box_file)
    return box_df
