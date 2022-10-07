import random
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sacred import SETTINGS
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from utils import loggers

# Settings
SETTINGS.DISCOVER_SOURCES = "sys"
SETTINGS.DISCOVER_DEPENDENCIES = "sys"

# Paths
FOLD_PATH = "data/NF/split.csv"         # data list and split
TEST_PATH = "data/NF/split_test.csv"    # data list and split for testing
BBOX_PATH = "data/NF/nf_box.csv"        # bounding box list, negative coordinates means using the whole image


def setup(ex):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # Define configurations
    @ex.config
    def configurations():
        # global
        observer = "file_storage"       # str, which observer to use. 'mongo' requires `pymongo` library. [none|mongo|file_storage]
        if observer == "mongo":
            host = "localhost"          # str, MongoDB host address
            port = 7000                 # int, MongoDB port
        tag = "default"                 # str, Configuration tag
        logdir = "logs"                 # str, Directory to save checkpoint and logs

        # model
        model = "din"                   # str, model name
        init_channel = 30               # int, initialize number of channels of DIN
        max_channels = 320              # int, maximum number of channels of DIN
        n_class = 2                     # str, number of output classes
        weight_decay = 3e-5             # float, Weight decay for regularizer
        gamma = 1.                      # float, scale of ExpDT

        # data loader
        fold = 0                        # int, test fold number, other folds are training folds
        depth = 10                      # int, volume depth, numbers of the scan planes
        height = 512                    # int, volume height, height of single scanning plane
        width = 160                     # int, volume width, width of single scanning plane
        channel = 1                     # int, volume channel, it should be fixed to 1 in this framework
        bs = 4                          # int, batch size for training
        zoom = (1.0, 1.25)              # tuple, random zoom range
        rotate = 5                      # float, random rotate angle, range (-rotate, rotate)
        flip = 7                        # int, composition of random flip flags, 1: left/right, 2: up/down, 4: front/back
        tumor_percent = 0.5             # float, minimum sample percentage in a batch containing tumors
        guide = "exp"                   # str, guide type [none|exp|euc|geo]
        exp_stddev = (1., 5., 5.)       # float, Stddev for ExpDT guide
        train_n = 1600                  # int, number of samples per training epoch
        val_n = train_n // 10           # int, Number of samples for evaluation
        test_depth = -1                 # int, test image depth, -1 means accepting various depths
        test_height = 960               # int, test image height, target height to resize
        test_width = 320                # int, test image width, target width to resize
        num_inters_max = 11             # int, maximum number of interaction points during training
        geo_lamb = 1.0                  # float, geodesic lambda, 1.0 for geo, 0.0 for euc
        geo_iter = 1                    # int, geodesic iteration

        # training
        epochs = 250                    # int, Number of epochs for training
        lr = 3e-4                       # float, Base learning rate for model training
        lrp = "plateau"                 # str, Learning rate policy [custom|period|poly|plateau]
        if lrp == "step":
            lr_boundaries = []          # list of int, [custom] boundaries to update lr
            lr_values = []              # list of float, [custom] updated lr values
        elif lrp == "period":
            lr_decay_step = 10          # float, [period] lr decay step
            lr_decay_rate = 0.1         # float, [period, plateau] lr decay rate
        elif lrp == "poly":
            lr_power = 0.9              # float, [poly] Polynomial power
            lr_end = 1e-6               # float, [poly, plateau] The minimal end learning rate
        elif lrp == "plateau":
            lr_patience = 30            # int, [plateau] Learning rate patience for decay
            lr_min_delta = 1e-4         # float, [plateau] Minimum delta to indicate improvement
            lr_decay_rate = 0.2         # float, [period_step, plateau] Learning rate decay rate
            lr_end = 1e-6               # float, [poly, plateau] The minimal end learning rate
            cool_down = 0               # int, [plateau]
            monitor = "val_loss"        # str, [plateau] Quantity to be monitored [val_loss/loss]
        optimizer = "adam"              # str, Optimizer for training [adam/momentum]
        if optimizer == "adam":
            beta_1 = 0.9                # float, [adam] Parameter
            beta_2 = 0.99               # float, [adam] Parameter
            epsilon = 1e-8              # float, [adam] Parameter
        elif optimizer == "momentum":
            momentum = 0.9              # float, [momentum] Parameter
            nesterov = False            # bool, [momentum] Parameter
        loss_w_type = "numerical"       # str, class weights [none|numerical]
        loss_w = (1, 3)                 # list, loss weight, one value for a class

        # testing
        inter_thresh = 0.85             # float, threshold of the segmentation target
        tta = True                      # bool, enable test time augmentation (flip)
        metrics = ["Dice", "VOE", "RVD"]    # list, test metric names. [Dice/VOE/RVD/ASSD/RMSD/MSD]
        save_predict = False            # bool, save prediction
        save_dir = "predict"            # str, directory to save prediction
        max_iter = 20                   # int, maximum iters for each image in interaction
        test_n = -1                     # int, number of samples to test, -1 means test all val set.
        test_set = "eval"               # str, test dataset [eval|test]
        use_box = False                 # bool, Use pre-defined boxes when evaluating

        # paths
        fold_path = FOLD_PATH
        bbox_path = BBOX_PATH
        resume_dir = ""                 # str, resume checkpoint from this directory
        if test_set == 'test':
            fold_path = TEST_PATH


    @ex.config_hook
    def config_hook(config, command_name, logger):
        # Create observers
        add_observers(ex, config, db_name=ex.path)
        ex.logger = loggers.get_global_logger(name=ex.path)

        # Limit gpu memory usage
        # all_gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in all_gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)

        return {}

    return ex


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot`
    operation. For example, you can access `cfg["split"]` by `cfg.split`, which
    makes the code more clear. Notice that the result object is a
    sacred.config.custom_containers.ReadOnlyDict, which is a read-only dict for
    preserving the configuration.

    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)


def add_observers(ex, config, db_name="default"):
    obs = config["observer"]
    if obs == "file_storage":
        observer_file = FileStorageObserver(config["logdir"])
        ex.observers.append(observer_file)
    elif obs == "mongo":
        observer_mongo = MongoObserver(url=f"{config['host']}:{config['port']}", db_name=db_name)
        ex.observers.append(observer_mongo)
    elif obs != "none":
        raise ValueError(f"`observer` only support [none|mongo|file_storage], got {abs}.")


def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    tf.random.set_seed(opt.seed)


def initialize(ex, run, config):
    opt = MapConfig(config)
    set_seed(opt)
    ex.logger.info(f"Run: {' '.join(sys.argv)}")

    logdir = Path(opt.logdir) / str(run._id)
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
    ex.logger.info(f"Log: {logdir.absolute()}")
    run.logdir_ = str(logdir)

    return opt, ex.logger
