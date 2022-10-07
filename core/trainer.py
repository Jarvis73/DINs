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

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import scipy.ndimage as ndi
from skimage.morphology import skeletonize, skeletonize_3d
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from core.metrics import get_metric_fns, MetricGroups, metric_3d, ConfusionMatrix
from core.losses import sample_weight
from core.solver import Solver
from data_kits.nf_kits import write_nii
from data_kits import np_ops
from utils.timer import Timer


class Trainer(object):
    def __init__(self, opt, logger, model: tf.keras.Model):
        self.opt = opt
        self.logger = logger
        self.model = model
        self.timer = Timer()
        self.num_classes = opt.n_class
        self.template_state = "[{:d}/{:d}] LR: {:g} [spd {:.1f} it/s]"
        self.start_epoch = [1]      # list is stateful and can be modified by ModelCheckpoint

        # build solver
        self.solver = Solver(opt)
        self.optimizer = self.solver.optim
        self.model.optimizer = self.optimizer
        # build loss function
        self.cross_entropy = SparseCategoricalCrossentropy(from_logits=True)
        # build metric functions
        self.train_loss_metric, self.eval_loss_metric, self.eval_acc_metric = get_metric_fns()
        self.callbacks = [self.solver.lr_callback]

        logger.info(" " * 11 + "==> Trainer")

    def record_metric(self, loss_metric, loss, acc_metrics, label, predictions):
        loss_metric(loss)
        if acc_metrics is not None:
            lab = tf.one_hot(label, self.num_classes)[..., 1]
            pred = tf.one_hot(tf.argmax(predictions, axis=-1), self.num_classes)[..., 1]
            acc_metrics(lab, pred)

    @tf.function
    def train_step(self, feature, label, weight=None):
        images = tf.where(tf.math.is_nan(feature[0]), tf.zeros_like(feature[0]), feature[0])
        check_op2 = tf.Assert(tf.reduce_all(tf.logical_not(tf.math.is_nan(feature[1]))),
                              ["Nan encountered in `guide map`", feature[1]])
        with tf.control_dependencies([check_op2]):
            feature = (tf.identity(images), tf.identity(feature[1]))
        with tf.GradientTape() as tape:
            predictions = self.model(feature, training=True)
            weight = sample_weight(
                label, self.num_classes, self.opt.loss_w_type, self.opt.loss_w)
            loss = self.cross_entropy(label, predictions, weight)
            loss += sum(self.model.losses)      # Add regularization losses
            check_op3 = tf.Assert(tf.reduce_all(tf.logical_not(tf.math.is_nan(loss))),
                                  ["Nan encountered in `loss`", loss])
            with tf.control_dependencies([check_op3]):
                loss = tf.identity(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.record_metric(self.train_loss_metric, loss, None, label, predictions)

    @tf.function
    def eval_step(self, feature, label):
        predictions = self.model(feature, training=False)
        loss = self.cross_entropy(label, predictions)
        self.record_metric(self.eval_loss_metric, loss, self.eval_acc_metric, label, predictions)

    def reset_states(self):
        self.train_loss_metric.reset_states()
        self.eval_loss_metric.reset_states()
        self.eval_acc_metric.reset_state()
        self.timer.reset()

    @staticmethod
    def log_str(flag, loss_metric, acc_metric):
        total = f"{flag} Loss: {loss_metric.result():.4f}"
        if acc_metric is not None:
            total += f" {acc_metric.name}: {acc_metric.result():.3f}"
        return total

    def start_training_loop(self, train_dataset, val_dataset):
        train_dataset, input_shape = train_dataset
        eval_dataset = val_dataset[0]

        for callback in self.callbacks:
            # Register model to callbacks
            callback.set_model(self.model)
            callback.on_train_begin()

        # Use input_shape to initialize all variables and pre-build computation graph
        self.model.build(input_shape)

        # Start training loop
        self.reset_states()
        for epoch in range(self.opt.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            for i, batch in enumerate(train_dataset):
                with self.timer.start():
                    for callback in self.callbacks:
                        callback.on_train_batch_begin(i)
                    self.train_step(*batch)
                    for callback in self.callbacks:
                        callback.on_train_batch_end(i)

            for feature, label in eval_dataset:
                self.eval_step(feature, label)

            self.logger.info(self.template_state.format(
                epoch, self.opt.epochs, self.optimizer.lr.numpy(), self.timer.cps))
            self.logger.info(self.log_str("    Train", self.train_loss_metric, None) + " " +
                             self.log_str("Val", self.eval_loss_metric, self.eval_acc_metric))

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, {
                    "loss": self.train_loss_metric.result().numpy(),
                    "val_loss": self.eval_loss_metric.result().numpy(),
                    "learning_rate": self.optimizer.lr.numpy(),
                    self.eval_acc_metric.name: self.eval_acc_metric.result().numpy()
                })
            if epoch != self.opt.epochs - 1:
                self.reset_states()

        for callback in self.callbacks:
            callback.on_train_end()

        return (
            self.train_loss_metric.result().numpy(),
            None,
            self.eval_loss_metric.result().numpy(),
            self.eval_acc_metric.result().numpy()
        )


################################################################################################
#
#   Evaluators 2D/3D and helper functions
#
################################################################################################


def inter_simulation_test(pred, ref, ndim=2, memory=None):
    """
    Interaction simulation, including positive points and negative points in test time
    Support 2d/3d mode

    1. Compute false positive and false negative areas
    2. Connectivity analysis
    3. Choice the largest error region
    4. Please a new click in the center of the error region
    5. Determine foreground or background pixel

    Parameters
    ----------
    pred: np.ndarray, 2d/3d, prediction
    ref:  np.ndarray, 2d/3d, reference
    ndim: int, 2/3, dimension
    memory: None or a list of points, store last points

    Returns
    -------
    pos: list, a list of two values (y, x) / (z, y, x)
    fg: 0 for positive guide, 1 for negative
    """
    pred = np.asarray(pred, np.bool)
    ref = np.asarray(ref, np.bool)
    sym_diff = pred ^ ref
    struct = ndi.generate_binary_structure(ndim, 1)
    res, n_obj = ndi.label(sym_diff, struct)
    counts = np.bincount(res.flat)
    args = np.argsort(counts[1:]) + 1

    def get_pos(idx):
        area = np.stack(np.where(res == idx), axis=1)
        pt = np.mean(area, axis=0).round(0).astype(np.int32)
        if ndim == 2:
            if not sym_diff[pt[0], pt[1]]:
                ske = np.stack(np.where(skeletonize(sym_diff)), axis=1)
                min_i = np.argmin(np.sum((ske - pt) ** 2, axis=1))
                pt = ske[min_i]
            fg = 0 if ref[pt[0], pt[1]] else 1
        else:
            if not sym_diff[pt[0], pt[1], pt[2]]:
                try:
                    ske = np.stack(np.where(skeletonize_3d(sym_diff)), axis=1)
                    min_i = np.argmin(np.sum((ske - pt) ** 2, axis=1))
                    pt = ske[min_i]
                except ValueError:
                    z_id = np.argmax(sym_diff.sum(axis=(1, 2)))
                    ske = np.stack(np.where(skeletonize(sym_diff[z_id])), axis=1)
                    min_i = np.argmin(np.sum((ske - pt[1:]) ** 2, axis=1))
                    pt = np.array([z_id, ske[min_i, 0], ske[min_i, 1]], np.int32)
            fg = 0 if ref[pt[0], pt[1], pt[2]] else 1
        return pt, fg

    def del_pos(p):
        if ndim == 2:
            sym_diff[p[0], p[1]] = 0
        else:
            sym_diff[p[0], p[1], p[2]] = 0

    index = 1
    while True:
        if n_obj == 0:    # no more objects
            return None, None
        elif n_obj == 1:    # Only one object to select
            point, is_fg = get_pos(args[-1])
            if memory is not None and np.any(np.all(np.isclose(point, memory), axis=1)):
                del_pos(point)
            else:
                break
        else:               # More selection
            point, is_fg = get_pos(args[-index])
            # Check whether is a new point
            if memory is not None and np.any(np.all(np.isclose(point, memory), axis=1)):
                index += 1  # Check for next connected component
                if index > n_obj:
                    point, is_fg = get_pos(args[-1])
                    del_pos(point)
                    point, is_fg = get_pos(args[-1])
                    break
            else:
                break
    return point.tolist(), is_fg


def geo_guide(pts, image, ctr_zoom_scale, opt):
    if len(pts) > 0:
        # Down-sample for acceleration
        int_ctr = (np.array(pts, np.float32).reshape(-1, 2) * ctr_zoom_scale / [1, 2, 2]).astype(np.int32)
        gd = np_ops.gen_guide_geo_3d(image, int_ctr, opt.geo_lamb, opt.geo_iter)
    else:
        gd = np.zeros_like(image, np.float32)
    return gd


def update_guide_simul(pred, ref, guide, opt, iteration, pos_col, ndim, img=None, ctr_zoom_scale=None):
    """ Add an interactive point by simulation.

    Parameters
    ----------
    pred: np.ndarray
        with shape [height, width] / [depth, height, width]
    ref: np.ndarray
        with shape [height, width] / [depth, height, width]
    guide: np.ndarray
        with shape [height, width] / [depth, height, width]
    opt:
        configurations
    iteration:
        a list of two integers, accumulate interactions of this case
    pos_col:
        [[...], [...]], accumulate points of interactions.
        FG points are in the first list and BG points are in the second.
    ndim: int
        2 or 3, dimension
    img: np.ndarray
        with shape [test_height, test_width] / [test_depth, test_height, test_width],
        for computing geodesic guide
    ctr_zoom_scale: np.ndarray
        with shape [2] / [3], for computing geodesic guide

    Returns
    -------
    guide: np.ndarray, updated parameter with the same shape
    pos: list, coordinates of the interactive point
    fg: int, 0/1, 0 for fg and 1 for bg
    pos_col: [[...], [...]], updated parameter
    """
    # Initialize an empty pred
    if pred is None:
        pred = np.zeros_like(ref, dtype=np.uint8)
    memory = np.concatenate((np.array(pos_col[0]).reshape(-1, ndim), np.array(pos_col[1]).reshape(-1, ndim)), axis=0) \
        if len(pos_col[0]) > 0 or len(pos_col[1]) > 0 else None
    pos, fg = inter_simulation_test(pred, ref, ndim, memory)
    if pos is None:
        return guide, None, None, pos_col
    pos_col[fg].append(pos)

    if opt.guide == "exp":
        cur_guide = np_ops.gen_guide_nd(
            ref.shape, np.array([pos]), np.ones((1, ndim), np.float32) * opt.exp_stddev, euclidean=False)
        update_op = np.maximum
    elif opt.guide == "euc":
        cur_guide = np_ops.gen_guide_nd(
            ref.shape, np.array([pos]), euclidean=True)
        update_op = np.minimum
    elif opt.guide == "geo":
        if ndim != 3:
            raise ValueError(f"Unsupported {ndim}D image for geodestic. [2]")
        down_img = img[:, ::2, ::2]
        fg_gd = geo_guide(pos_col[0], down_img, ctr_zoom_scale, opt)
        bg_gd = geo_guide(pos_col[1], down_img, ctr_zoom_scale, opt)
        guide = np.stack((fg_gd, bg_gd), axis=-1)
        zoom_scale = np.array(img.shape + (1,), np.float32) / np.array(fg_gd.shape + (1,), np.float32)
        guide = ndi.zoom(guide, zoom_scale, order=1)
    else:
        raise ValueError(f"Unsupported guide type: {opt.guide}. [none/exp/euc/geo]")
    if opt.guide in ["exp", "euc"]:
        if guide is None:
            guide = np.zeros(ref.shape + (2,), dtype=np.float32)
        guide[..., fg] = update_op(guide[..., fg], cur_guide) if guide[..., fg].max() > 0 else cur_guide
    iteration[fg] += 1
    return guide, pos, fg, pos_col


def compute_dice(pred, ref):
    inter = np.count_nonzero(pred * ref)
    union = np.count_nonzero(pred) + np.count_nonzero(ref)
    return (2 * inter + 1e-8) / (union + 1e-8)


class Evaluator2D(object):
    def __init__(self, opt, logger, model):
        """ Evaluator class for 2D models """
        self.opt = opt
        self.logger = logger
        self.model = model
        self.timer = Timer()
        self.num_classes = opt.n_class
        # build metric functions
        self.eval_metrics = MetricGroups(["NF"])
        self.tta_dict = {
            1: [2], 2: [1], 3: [1, 2]
        }
        if self.opt.tf_func:
            self.eval_step = tf.function(self.eval_step)
        self.eval_ops = self.eval_tta if self.opt.tta else self.eval_step

    def reset_states(self):
        self.eval_metrics.reset()
        self.timer.reset()

    def record_metric(self, label, predictions):
        lab = np_ops.one_hot(label, self.num_classes, axis=0)[1]
        pred = np_ops.one_hot(predictions, self.num_classes, axis=0)[1]
        cls_metrics = metric_3d(pred, lab, metrics_eval=self.opt.metrics)
        conf = ConfusionMatrix(pred, lab)
        conf.compute()
        cls_metrics.update({"fn": conf.fn, "fp": conf.fp, "tp": conf.tp})
        total = {"NF": cls_metrics}
        self.eval_metrics.add(total)
        return total.copy()

    def load_weights(self, input_shape):
        if not (Path(self.opt.resume_dir) / "checkpoint").exists():
            raise FileNotFoundError(f'Checkpoint file not found: {Path(self.opt.resume_dir) / "checkpoint"}')
        self.model.build(input_shape)
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_path = tf.train.latest_checkpoint(self.opt.resume_dir)
        ckpt.restore(ckpt_path).expect_partial()
        self.logger.info(' ' * 11 + f"==> Restore checkpoint from {ckpt_path}")

    @tf.function
    def eval_step(self, feature):
        if isinstance(feature, list) and len(feature) == 1:
            feature = feature[0]
        predictions = self.model(feature, training=False)
        probability = tf.nn.softmax(predictions, axis=-1)
        return probability

    def eval_tta(self, feature):
        probs = [self.eval_step(feature)]
        for i in range(1, 4):
            if self.opt.flip & i > 0:
                axes = self.tta_dict[i]
                new_feature = [tf.reverse(im, axes) for im in feature]
                probs.append(tf.reverse(self.eval_step(new_feature), axes))
        avg_prob = sum(probs) / len(probs)
        return avg_prob

    def eval_auto(self, volume_gen, label):
        """ Evaluate with interactions by simulation.

        Parameters
        ----------
        volume_gen: generator, produce (int, np.ndarray), shape [test_height, test_width, channel]
        label: np.ndarray, with shape [ori_depth, ori_height, ori_width]

        Returns
        -------
        final_pred: np.ndarray, final prediction of this case
        """
        _, height, width = label.shape
        final_pred = np.zeros_like(label, dtype=np.uint8)
        for si, img in volume_gen:
            feature = [tf.convert_to_tensor(img[None], tf.float32)]
            pred = self.eval_ops(feature)     # [test_height, test_width]
            pred = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)[0]
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
            final_pred[si] = pred
        return final_pred

    def eval_inter_simul(self, volume_gen, label, sample):
        """ Evaluate with interactions by simulation.

        Parameters
        ----------
        volume_gen: generator, produce (int, np.ndarray), shape [test_height, test_width, channel]
        label: np.ndarray, with shape [ori_depth, ori_height, ori_width]
        sample: pd.Series, use `sample.pid` to get patient id

        Returns
        -------
        final_pred: np.ndarray, final prediction of this case
        """
        ori_shape_cv = tuple(label.shape[:0:-1])
        run_shape_cv = (self.opt.test_width, self.opt.test_height)
        final_pred = np.zeros_like(label, dtype=np.uint8)
        case_inters = [0, 0]
        for si, img in volume_gen:
            print("(Case {:3d} Slice {:2d}) Interacting: ".format(int(sample.pid), int(si)), end="", flush=True)
            feature = [tf.convert_to_tensor(img[None], tf.float32), None]
            guide, pred = None, None
            num_iter, last_pos = [0, 0], None
            pos_col = [[], []]      # Temporarily not used
            while True:
                guide, new_pos, fg, pos_col = update_guide_simul(
                    pred, label[si], guide, self.opt, num_iter, pos_col, ndim=2)
                last_pos = new_pos
                resized_guide = cv2.resize(guide, run_shape_cv, interpolation=cv2.INTER_LINEAR)
                feature[1] = tf.convert_to_tensor(resized_guide[None], tf.float32)
                # (Soft result) Prediction from networks
                pred = self.eval_ops(feature)     # [test_height, test_width]
                pred = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)[0]
                pred = cv2.resize(pred, ori_shape_cv, interpolation=cv2.INTER_NEAREST)
                # (Hard result) Add user interaction points to pred
                for p in pos_col[0]:
                    pred[p[0], p[1]] = 1
                for p in pos_col[1]:
                    pred[p[0], p[1]] = 0
                dice = compute_dice(pred, label[si])
                print("{:.3f}->".format(dice), end="", flush=True)
                # Reach the dice threshold or threshold of the number of the interactions
                if dice >= self.opt.inter_thresh or num_iter[0] + num_iter[1] >= self.opt.max_iter:
                    print(" Number iters: {}/{}".format(num_iter[0], num_iter[1]))
                    case_inters[0] += num_iter[0]
                    case_inters[1] += num_iter[1]
                    break
            final_pred[si] = pred
        return final_pred, case_inters

    def start_evaluating_loop(self, eval_dataset):
        eval_dataset, input_shape = eval_dataset
        self.load_weights(input_shape)
        self.reset_states()
        total_inters = [0, 0]
        if self.opt.save_dir:
            save_dir = Path(self.opt.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for sample, meta_data, volume_gen, label in eval_dataset:
            with self.timer.start():
                final_pred, case_inters = self.eval_inter_simul(volume_gen, label, sample)
                total_inters[0] += case_inters[0]
                total_inters[1] += case_inters[1]
                case_metrics = self.record_metric(label, final_pred)

            # Save prediction
            if self.opt.save_dir:
                save_file = save_dir / f"predict-{sample.pid}.nii.gz"
                write_nii(final_pred, meta_data, save_file)
            self.logger.info(
                f"Evaluate {sample.pid} - Elapse {self.timer.diff:.1f}s" +
                " (Saved)" * bool(len(self.opt.save_dir))
            )
            for cls, metrics in case_metrics.items():
                self.logger.info(f"    {cls} ==> " + "".join(["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]))

        self.logger.info(
            f"Total infer {self.timer.calls} cases"
            + f" Elapse {self.timer.acc_time:.1f}s"
            + f" {self.timer.spc:.1f} s/it"
        )
        for cls, metrics in self.eval_metrics.result().items():
            tp, fp, fn = metrics.pop("tp"), metrics.pop("fp"), metrics.pop("fn")
            lst = ["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]
            lst.append("G_Dice: {:.3f} ".format(2 * tp / (2 * tp + fn + fp)))
            lst.append("(Avg inters: {:.1f}/{:.1f})".format(
                total_inters[0] / self.timer.calls, total_inters[1] / self.timer.calls))
            self.logger.info(f"    {cls} ==> " + "".join(lst))

        return


class Evaluator3D(object):
    def __init__(self, opt, logger, model, run):
        self.opt = opt
        self.logger = logger
        self.model = model
        self.logdir = Path(run.logdir_)
        self.timer = Timer()
        self.num_classes = opt.n_class
        # build metric functions
        self.eval_metrics = MetricGroups(["NF"])
        self.eval_ops = self.eval_tta if self.opt.tta else self.eval_step

        self.tta_dict = {
            1: [3], 2: [2], 3: [2, 3], 4: [1],
            5: [1, 3], 6: [1, 2], 7: [1, 2, 3]
        }

        logger.info(" " * 11 + "==> Evaluator3D")

    def reset_states(self):
        self.eval_metrics.reset()
        self.timer.reset()

    def record_metric(self, label, predictions):
        lab = np_ops.one_hot(label, self.num_classes, axis=0)[1]
        pred = np_ops.one_hot(predictions, self.num_classes, axis=0)[1]
        cls_metrics = metric_3d(pred, lab, metrics_eval=self.opt.metrics)
        conf = ConfusionMatrix(pred, lab)
        conf.compute()
        cls_metrics.update({"fn": conf.fn, "fp": conf.fp, "tp": conf.tp})
        total = {"NF": cls_metrics}
        self.eval_metrics.add(total)
        return total.copy()

    def load_weights(self, input_shape):
        if not (Path(self.opt.resume_dir) / "checkpoint").exists():
            raise FileNotFoundError(f'Checkpoint file not found: {Path(self.opt.resume_dir) / "checkpoint"}')
        self.model.build(input_shape)
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_path = tf.train.latest_checkpoint(self.opt.resume_dir)
        ckpt.restore(ckpt_path).expect_partial()
        self.logger.info(' ' * 11 + f"==> Restore checkpoint from {ckpt_path}")

    @tf.function
    def eval_step(self, feature):
        if isinstance(feature, list) and len(feature) == 1:
            feature = feature[0]
        predictions = self.model(feature, training=False)
        probability = tf.nn.softmax(predictions, axis=-1)
        return probability

    def eval_tta(self, feature):
        probs = [self.eval_step(feature)]
        for i in range(1, 8):
            if self.opt.flip & i > 0:
                axes = self.tta_dict[i]
                new_feature = [tf.reverse(im, axes) for im in feature]
                probs.append(tf.reverse(self.eval_step(new_feature), axes))
        avg_prob = sum(probs) / len(probs)
        return avg_prob

    def eval_auto(self, volume, label):
        """ Evaluate automatically (without guides)

        Parameters
        ----------
        volume: np.ndarray, with shape [test_height, test_width, channel]
        label: np.ndarray, with shape [ori_depth, ori_height, ori_width]

        Returns
        -------
        final_pred: np.ndarray, final prediction of this case
        """
        zoom_scale = np.array([1,
                               label.shape[1] / volume.shape[1],
                               label.shape[2] / volume.shape[2]], np.float32)
        feature = [tf.convert_to_tensor(volume[None, ..., None], tf.float32)]
        pred = self.eval_ops(feature)     # [test_depth, test_height, test_width]
        pred = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)[0]
        final_pred = ndi.zoom(pred, zoom_scale, order=0)
        if final_pred.shape[0] != label.shape[0]:
            final_pred = final_pred[:label.shape[0] - final_pred.shape[0]]
        return final_pred

    def eval_inter_simul(self, volume, label, sample):
        """ Evaluate with interactions by simulation.

        Parameters
        ----------
        volume: np.ndarray, with shape [test_depth, test_height, test_width]
        label: np.ndarray, with shape [ori_depth, ori_height, ori_width]
        sample: pd.Series, use `sample.pid` to get patient id

        Returns
        -------
        final_pred: np.ndarray, final prediction of this case
        """
        zoom_scale = np.array([1,
                               volume.shape[1] / label.shape[1],
                               volume.shape[2] / label.shape[2],
                               1], np.float32)
        label_bool = np.asarray(label, np.bool)
        if label.shape[0] < volume.shape[0]:
            label_bool = np.pad(label, ((0, volume.shape[0] - label.shape[0]), (0, 0), (0, 0)),
                                mode="constant", constant_values=0)
        case_inters = [0, 0]
        print("(Case {:3d}) Interacting: ".format(int(sample.pid)), end="", flush=True)
        feature = [tf.convert_to_tensor(volume[None, ..., None], tf.float32), None]
        guide, pred = None, None
        num_iter = [0, 0]
        pos_col = [[], []]      # Temporarily not used
        while True:
            geo_kwargs = {"img": volume, "ctr_zoom_scale": zoom_scale[:-1]} \
                if "geo" in self.opt.guide else {}
            guide, new_pos, fg, pos_col = update_guide_simul(
                pred, label_bool, guide, self.opt, num_iter, pos_col, ndim=3, **geo_kwargs)
            resized_guide = guide.copy() \
                if "geo" in self.opt.guide else ndi.zoom(guide, zoom_scale, order=1)
            feature[1] = tf.convert_to_tensor(resized_guide[None] * self.opt.gamma, tf.float32)
            pred = self.eval_ops(feature)     # [test_height, test_width]
            pred = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)[0]
            pred = ndi.zoom(pred, 1. / zoom_scale[:-1], order=0)
            dice = compute_dice(pred, label_bool)
            print("{:.3f}->".format(dice), end="", flush=True)
            # Reach the dice threshold or threshold of the number of the interactions
            if dice >= self.opt.inter_thresh or num_iter[0] + num_iter[1] >= self.opt.max_iter:
                print(" Number iters: {}/{}".format(num_iter[0], num_iter[1]))
                case_inters[0] += num_iter[0]
                case_inters[1] += num_iter[1]
                break
        if label.shape[0] != volume.shape[0]:
            pred = pred[:label.shape[0] - volume.shape[0]]
        return pred, case_inters

    def start_evaluating_loop(self, eval_dataset):
        opt = self.opt
        eval_dataset, input_shape = eval_dataset
        self.load_weights(input_shape)
        self.reset_states()
        save_dir = self.logdir / opt.save_dir
        if opt.save_predict:
            save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'-' * 50)
        self.logger.info(f'Target thresh: {opt.inter_thresh}')
        self.logger.info(f'Max simulated interactions: {opt.max_iter}')
        self.logger.info(f'Use predefined bounding boxes: {opt.use_box}')
        self.logger.info(f'Enable test-time augmentation (tta): {opt.tta}')
        self.logger.info(f'Saving directory: {save_dir if opt.save_predict else "[Disabled]"}')
        self.logger.info(f'Test set: {opt.test_set}')
        self.logger.info(f'Number of samples: {opt.test_n if opt.test_n >= 0 else "[ALL]"}')
        self.logger.info(f'-' * 50)
        self.logger.info(f'Start testing...')
        total_inters = [0, 0]
        for sample, meta_data, volume, label in eval_dataset:
            self.timer.tic()
            final_pred, case_inters = self.eval_inter_simul(volume, label, sample)
            total_inters[0] += case_inters[0]
            total_inters[1] += case_inters[1]
            case_metrics = self.record_metric(label, final_pred)
            self.timer.toc()

            # Save prediction
            if opt.save_predict:
                save_file = save_dir / f"predict-{int(sample.pid)}.nii.gz"
                write_nii(final_pred, meta_data, save_file)
            self.logger.info(
                f"Evaluate {int(sample.pid)}"
                + " - Elapse {:.1f}s".format(self.timer.diff)
                + " - (Saved)" * opt.save_predict
            )
            for cls, metrics in case_metrics.items():
                self.logger.info(f"    {cls} ==> " + "".join(["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]))

        self.logger.info(
            f"Total infer {self.timer.calls} cases"
            + " - Elapse {:.1f}s".format(self.timer.total_time)
            + " - {:.1f} s/it".format(self.timer.spc)
        )
        for cls, metrics in self.eval_metrics.result().items():
            tp, fp, fn = metrics.pop("tp"), metrics.pop("fp"), metrics.pop("fn")
            lst = ["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]
            lst.append("G_Dice: {:.3f}".format(2 * tp / (2 * tp + fn + fp)))
            lst.append("(Avg inters: {:.1f}/{:.1f})".format(
                total_inters[0] / self.timer.calls, total_inters[1] / self.timer.calls))
            self.logger.info(f"    {cls} ==> " + "".join(lst))


class Evaluator3DWithBox(Evaluator3D):
    def __init__(self, opt, logger, model, run):
        """ Evaluator class for 3D models. Boxes are used during retriving data """
        super(Evaluator3DWithBox, self).__init__(opt, logger, model, run)
        self.tta_dict = {
            1: [3], 2: [2], 3: [2, 3], 4: [1],
            5: [1, 3], 6: [1, 2], 7: [1, 2, 3]
        }

        logger.info(f"           ==> Evaluator3DWithBox")

    def eval_inter_simul_with_box(self, volume, label, max_iter, data_mask, final_pred, final_label,
                                  crop_box):
        """ Evaluate with interactions by simulation.

        Parameters
        ----------
        volume: np.ndarray, with shape [?, ?, ?]
        label: np.ndarray, with shape [?, ?, ?]
        sample: pd.Series, use `sample.pid` to get patient id
        max_iter:
        data_mask: np.ndarray, the same shape with volume
        final_pred:
        final_label:
        crop_box:

        Returns
        -------
        final_pred: np.ndarray, final prediction of this case
        """
        zoom_scale = np.array([1,
                               volume.shape[1] / label.shape[1],
                               volume.shape[2] / label.shape[2],
                               1], np.float32)
        label_bool = np.asarray(label, np.bool)
        if label.shape[0] < volume.shape[0]:
            label_bool = np.pad(label, ((0, volume.shape[0] - label.shape[0]), (0, 0), (0, 0)),
                                mode="constant", constant_values=0)
            if data_mask is not None:
                data_mask = np.pad(data_mask, ((0, volume.shape[0] - label.shape[0]), (0, 0), (0, 0)),
                                   mode="constant", constant_values=0)
        feature = [tf.convert_to_tensor(volume[None, ..., None], tf.float32), None]
        guide, pred = None, None
        num_iter = [0, 0]
        pos_col = [[], []]      # Temporarily not used
        data_save_counter = 0
        if crop_box is not None:
            final_pred_backup = final_pred[crop_box].copy()
        while True:
            geo_kwargs = {"img": volume, "ctr_zoom_scale": zoom_scale[:-1]} \
                if "geo" in self.opt.guide else {}
            guide, new_pos, fg, pos_col = update_guide_simul(
                pred, label_bool, guide, self.opt, num_iter, pos_col, ndim=3, **geo_kwargs)
            resized_guide = guide.copy() \
                if "geo" in self.opt.guide else ndi.zoom(guide, zoom_scale, order=1)
            feature[1] = tf.convert_to_tensor(resized_guide[None] * self.opt.gamma, tf.float32)
            pred = self.eval_ops(feature)     # [test_height, test_width]
            pred = tf.argmax(pred, axis=-1).numpy().astype(np.uint8)[0]
            if data_mask is not None:
                pred = pred * data_mask
            pred = ndi.zoom(pred, 1. / zoom_scale[:-1], order=0)
            if label.shape[0] != volume.shape[0]:
                patch_pred = pred[:label.shape[0] - volume.shape[0]]
            else:
                patch_pred = pred
            if crop_box is not None:
                final_pred[crop_box] = np.maximum(final_pred_backup, patch_pred)
            else:
                final_pred = patch_pred
            dice = compute_dice(final_pred, final_label)
            print("{:.3f}->".format(dice), end="", flush=True)
            # Reach the dice threshold or threshold of the number of the interactions
            if dice >= self.opt.inter_thresh or num_iter[0] + num_iter[1] >= max_iter:
                break
        return patch_pred, num_iter

    def start_evaluating_loop(self, eval_dataset):
        opt = self.opt
        eval_dataset, input_shape = eval_dataset
        self.load_weights(input_shape)
        self.reset_states()
        save_dir = self.logdir / opt.save_dir
        if opt.save_predict:
            save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'-' * 50)
        self.logger.info(f'Target thresh: {opt.inter_thresh}')
        self.logger.info(f'Max simulated interactions: {opt.max_iter}')
        self.logger.info(f'Use predefined bounding boxes: {opt.use_box}')
        self.logger.info(f'Enable test-time augmentation (tta): {opt.tta}')
        self.logger.info(f'Saving directory: {save_dir if opt.save_predict else "[Disabled]"}')
        self.logger.info(f'Test set: {opt.test_set}')
        self.logger.info(f'Number of samples: {opt.test_n if opt.test_n >= 0 else "[ALL]"}')
        self.logger.info(f'-' * 50)
        self.logger.info(f'Start testing...')
        total_inters = [0, 0]
        finish_case = False
        final_pred = None
        case_inters = [0, 0]
        self.timer.tic()
        for sample, meta_data, volume, label, crop_box, data_mask, max_iter in eval_dataset:
            lab_patch, label = label
            if final_pred is None:
                print("(Case {:3d}) Interacting: ".format(int(sample.pid)), end="", flush=True)
                final_pred = np.zeros(meta_data['dim'][3:0:-1], np.uint8)
            if crop_box is not None:
                bid, total, crop_box = crop_box
            patch_pred, patch_iter = self.eval_inter_simul_with_box(
                volume, lab_patch, max_iter, data_mask, final_pred, label, crop_box)
            case_inters[0] += patch_iter[0]
            case_inters[1] += patch_iter[1]
            if crop_box is None:
                # Full image inference
                finish_case = True
                final_pred = patch_pred
            else:
                if bid == total - 1:
                    finish_case = True
                # Image patch inference
                final_pred[crop_box] = np.maximum(final_pred[crop_box], patch_pred)

            if finish_case:
                print(" Number iters: {}/{}".format(case_inters[0], case_inters[1]))
                total_inters[0] += case_inters[0]
                total_inters[1] += case_inters[1]
                case_metrics = self.record_metric(label, final_pred)
                self.timer.toc()

                # Save prediction
                if self.opt.save_predict:
                    save_file = save_dir / f"predict-{int(sample.pid)}.nii.gz"
                    write_nii(final_pred, meta_data, save_file)
                self.logger.info(
                    f"Evaluate {int(sample.pid)}"
                    + " - Elapse {:.1f}s".format(self.timer.diff)
                    + " - (Saved)" * self.opt.save_predict
                )
                for cls, metrics in case_metrics.items():
                    self.logger.info(f"    {cls} ==> " + "".join(["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]))
                finish_case = False
                final_pred = None
                case_inters = [0, 0]
                self.timer.tic()

        self.logger.info(
            f"Total infer {self.timer.calls} cases"
            + " - Elapse {:.1f}s".format(self.timer.total_time)
            + " - {:.1f} s/it".format(self.timer.spc)
        )
        for cls, metrics in self.eval_metrics.result().items():
            tp, fp, fn = metrics.pop("tp"), metrics.pop("fp"), metrics.pop("fn")
            lst = ["{}: {:.3f} ".format(k, v) for k, v in metrics.items()]
            lst.append("G_Dice: {:.3f}".format(2 * tp / (2 * tp + fn + fp)))
            lst.append("(Avg inters: {:.1f}/{:.1f})".format(
                total_inters[0] / self.timer.calls, total_inters[1] / self.timer.calls))
            self.logger.info(f"    {cls} ==> " + "".join(lst))

        return
