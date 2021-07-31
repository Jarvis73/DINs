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

from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.keras.metrics import MeanMetricWrapper
from medpy import metric as mtr     # pip install medpy
from sacred import Ingredient


def lm_arguments():
    """ ==> Loss/Metrics Arguments"""
    loss_type = "xentropy"      # str, Loss type. [xentropy]
    loss_weight_type = "none"   # str, Weights for alleviating class imbalance problem. [none/numerical]
    loss_numeric_w = []         # list, [numerical] One value per class.
    train_metric = "none"      # str, Training metric name used during training. [Dice/VOE/RVD/IoU]
    eval_metric = "Dice"       # str, Evaluation metric name. [Dice/VOE/RVD/IoU]


@loss_metric_ingredient.capture
def get_loss_fn(loss_type):
    if loss_type == "xentropy":
        return K.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. [xentropy]")


def sample_weight(ground_truth, num_classes, loss_weight_type, loss_numeric_w):
    if loss_weight_type == "none":
        return None
    elif loss_weight_type == "numerical":
        ndim = len(ground_truth.shape)
        size = tf.cast(ground_truth.shape, tf.float32)
        assert num_classes == len(loss_numeric_w), f"{num_classes} vs {len(loss_numeric_w)}"
        one_hot = tf.one_hot(ground_truth, num_classes, axis=-1)
        w = tf.constant([loss_numeric_w for _ in range(ground_truth.shape[0])], dtype=tf.float32)
        if ndim == 3:
            w = tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, axis=1), axis=1) * one_hot, axis=-1)
        elif ndim == 4:
            w = tf.reduce_sum(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(w, axis=1), axis=1), axis=1) * one_hot, axis=-1)
        else:
            raise ValueError(f"Unsupported ground truth dimension: {ground_truth.shape}")
    else:
        raise ValueError(f"Unsupported loss weight type: {loss_weight_type}. [none/numerical]")

    w = w / tf.reduce_sum(w, axis=list(range(1, ndim)), keepdims=True) * \
        (size[1] * size[2] if ndim == 3 else size[1] * size[2] * size[3])
    return w
