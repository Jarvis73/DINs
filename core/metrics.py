from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.python.keras.metrics import MeanMetricWrapper
from medpy import metric as mtr     # pip install medpy

from utils.surface import Surface


def get_metric_fns():
    train_loss = Mean(name="train_loss")
    eval_loss = Mean(name="eval_loss")
    eval_acc = Dice(name="dice")
    return train_loss, eval_loss, eval_acc


class MetricGroup(object):
    def __init__(self):
        self.group = defaultdict(list)

    def add(self, new_metrics: dict):
        for k, v in new_metrics.items():
            self.group[k].append(v)

    def result(self):
        return {k: float(np.mean(v)) for k, v in self.group.items()}

    def reset(self):
        for v in self.group.values():
            v.clear()


class MetricGroups(object):
    def __init__(self, keys):
        self.group = {key: MetricGroup() for key in keys}

    def add(self, new_metrics: dict):
        for k, v in new_metrics.items():
            self.group[k].add(v)

    def add_key(self, key, new_metrics):
        self.group[key].add(new_metrics)

    def result(self):
        return {k: v.result() for k, v in self.group.items()}

    def reset(self):
        for v in self.group.values():
            v.reset()


class Dice(MeanMetricWrapper):
    def __init__(self, name="dice", dtype=None, eps=1e-7):
        super(Dice, self).__init__(metric_dice, name, dtype=dtype, eps=eps)


class VOE(MeanMetricWrapper):
    def __init__(self, name="voe", dtype=None, eps=1e-7):
        super(VOE, self).__init__(metric_voe, name, dtype=dtype, eps=eps)


class RVD(MeanMetricWrapper):
    def __init__(self, name="rvd", dtype=None, eps=1e-7):
        super(RVD, self).__init__(metric_rvd, name, dtype=dtype, eps=eps)


def metric_dice(label, prediction, eps=1e-7):
    """ Dice coefficient for N-D Tensor.
    Support "soft dice", which means logits and labels don't need to be binary tensors.

    Parameters
    ----------
    label: tf.Tensor, require tf.float32, shape [batch_size, ...]
    prediction: tf.Tensor, require tf.float32, shape [batch_size, ...]
    eps: float, epsilon is set to avoid dividing zero

    Returns
    -------
    dice: tf.Tensor, shape [batch_size], average on batch axis
    """
    assert label.shape == prediction.shape, f"{label.shape} vs {prediction.shape}"
    dim = len(label.shape)
    sum_axis = list(range(1, dim))
    intersection = tf.reduce_sum(label * prediction, axis=sum_axis)
    left = tf.reduce_sum(label, axis=sum_axis)
    right = tf.reduce_sum(prediction, axis=sum_axis)
    dice = (2 * intersection + eps) / (left + right + eps)
    return dice


def metric_voe(label, prediction, eps=1e-7):
    """ Volumetric Overlap Error for N-D Tensor.

    Parameters
    ----------
    label: tf.Tensor, require tf.float32, shape [batch_size, ...]
    prediction: tf.Tensor, require tf.float32, shape [batch_size, ...]
    eps: float, epsilon is set to avoid dividing zero

    Returns
    -------
    voe: tf.Tensor, shape [batch_size], average on batch axis
    """
    assert label.shape == prediction.shape, f"{label.shape} vs {prediction.shape}"
    dim = len(label.shape)
    sum_axis = list(range(1, dim))
    numerator = tf.reduce_sum(label * prediction, axis=sum_axis)
    denominator = tf.reduce_sum(tf.clip_by_value(label + prediction, 0.0, 1.0), axis=sum_axis)
    voe = 1.0 - numerator / (denominator + eps)
    return voe


def metric_rvd(label, prediction, eps=1e-7):
    """ Relative Absolute Volume Difference for N-D Tensor.

    Parameters
    ----------
    label: tf.Tensor, require tf.float32, shape [batch_size, ...]
    prediction: tf.Tensor, require tf.float32, shape [batch_size, ...]
    eps: float, epsilon is set to avoid dividing zero

    Returns
    -------
    rvd: tf.Tensor, shape [batch_size], average on batch axis
    """
    assert label.shape == prediction.shape, f"{label.shape} vs {prediction.shape}"
    dim = len(label.shape)
    sum_axis = list(range(1, dim))
    a = tf.reduce_sum(prediction, axis=sum_axis)
    b = tf.reduce_sum(label, axis=sum_axis)
    ravd = tf.abs(a - b) / (b + eps)
    return ravd


metric_iou = metric_voe
IoU = VOE


def metric_np_vd(label, prediction):
    """ Volume Difference for N-D numpy ndarray.

    Parameters
    ----------
    label: np.ndarray, require np.float32
    prediction: np.ndarray, require np.float32
    eps: float, epsilon is set to avoid dividing zero

    Returns
    -------
    vd: float
    """
    assert label.shape == prediction.shape, f"{label.shape} vs {prediction.shape}"
    return np.sum(label.astype(np.float32) - prediction.astype(np.float32))


def metric_np_rvd(label, prediction, eps=1e-7):
    """ Relative Volume Difference for N-D numpy ndarray.

    Parameters
    ----------
    label: np.ndarray, require np.float32
    prediction: np.ndarray, require np.float32
    eps: float, epsilon is set to avoid dividing zero

    Returns
    -------
    vd: float
    """
    assert label.shape == prediction.shape, f"{label.shape} vs {prediction.shape}"
    label, prediction = label.astype(np.float32), prediction.astype(np.float32)
    return np.sum(np.abs(label - prediction)) / (np.sum(label) + np.sum(prediction))


def metric_3d(logits3d, labels3d, metrics_eval=None, **kwargs):
    """
    Compute 3D metrics:

    * (Dice) Dice Coefficient

    * (VOE)  Volumetric Overlap Error

    * (RVD) Relative Absolute Volume Difference

    * (ASD)  Average Symmetric Surface Distance

    * (RMSD) Root Mean Square Symmetric Surface Distance

    * (MSD)  Maximum Symmetric Surface Distance

    * (VD)   Volume Difference

    * (RVD2)  Relative Volume Difference

    Parameters
    ----------
    logits3d: ndarray
        3D binary prediction, shape is the same with `labels3d`, it should be an int
        array or boolean array.
    labels3d: ndarray
        3D labels for segmentation, shape [None, None, None], it should be an int array
        or boolean array. If the dimensions of `logits3d` and `labels3d` are greater than
        3, then `np.squeeze` will be applied to remove extra single dimension and then
        please make sure these two variables are still have 3 dimensions. For example,
        shape [None, None, None, 1] or [1, None, None, None, 1] are allowed.
    metrics_eval: str or list
        a string or a list of string to specify which metrics need to be return, default
        this function will return all the metrics listed above. For example, if use
        ```python
        _metric_3D(logits3D, labels3D, metrics_eval=["Dice", "VOE", "ASD"])
        ```
        then only these three metrics will be returned.
    kwargs: dict
        sampling: list
            the pixel resolution or pixel size. This is entered as an n-vector where n
            is equal to the number of dimensions in the segmentation i.e. 2D or 3D. The
            default value is 1 which means pixls are 1x1x1 mm in size

    Returns
    -------
    metrics required

    Notes
    -----
    Thanks to the code snippet from @MLNotebook's blog.

    [Blog link](https://mlnotebook.github.io/post/surface-distance-function/).
    """
    required = metrics_eval
    metrics = ["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"] + ["VD", "RVD2"]
    need_dist_map = False

    if required is None:
        required = metrics
    elif isinstance(required, str):
        required = [required]
        if required[0] not in metrics:
            raise ValueError("Not supported metric: %s" % required[0])
        elif required in metrics[3:6]:
            need_dist_map = True
        else:
            need_dist_map = False

    for req in required:
        if req not in metrics:
            raise ValueError("Not supported metric: %s" % req)
        if (not need_dist_map) and req in metrics[3:6]:
            need_dist_map = True

    if logits3d.ndim > 3:
        logits3d = np.squeeze(logits3d)
    if labels3d.ndim > 3:
        labels3d = np.squeeze(labels3d)

    assert logits3d.shape == labels3d.shape, ("Shape mismatch of logits3D and labels3D. \n"
                                              "Logits3D has shape %r while labels3D has "
                                              "shape %r" % (logits3d.shape, labels3d.shape))
    logits3d = logits3d.astype(np.bool)
    labels3d = labels3d.astype(np.bool)

    metrics_3d = {}
    sampling = kwargs.get("sampling", [1., 1., 1.])

    if need_dist_map:
        if np.count_nonzero(logits3d) == 0 or np.count_nonzero(labels3d) == 0:
            metrics_3d['ASSD'] = 0
            metrics_3d['MSD'] = 0
        else:
            eval_surf = Surface(logits3d, labels3d, physical_voxel_spacing=sampling,
                                mask_offset=[0., 0., 0.],
                                reference_offset=[0., 0., 0.])

            if "ASSD" in required:
                metrics_3d["ASSD"] = eval_surf.get_average_symmetric_surface_distance()
                required.remove("ASSD")
            if "MSD" in required:
                metrics_3d["MSD"] = eval_surf.get_maximum_symmetric_surface_distance()
            if "RMSD" in required:
                metrics_3d["RMSD"] = eval_surf.get_root_mean_square_symmetric_surface_distance()

    if required:
        if "Dice" in required:
            metrics_3d["Dice"] = mtr.dc(logits3d, labels3d)
        if "VOE" in required:
            metrics_3d["VOE"] = 1. - mtr.jc(logits3d, labels3d)
        if "RVD" in required:
            metrics_3d["RVD"] = np.absolute(mtr.ravd(logits3d, labels3d))
        if "VD" in required:
            metrics_3d["VD"] = metric_np_vd(labels3d, logits3d)
        if "RVD2" in required:
            metrics_3d["RVD2"] = metric_np_rvd(labels3d, logits3d)

    return metrics_3d


class ConfusionMatrix(object):
    def __init__(self, test=None, reference=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")
        assert self.test.shape == self.reference.shape, "Shape mismatch: {} and {}".format(
            self.test.shape, self.reference.shape)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = self.reference.size
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full
