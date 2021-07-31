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

SETTINGS.DISCOVER_SOURCES = "sys"
SETTINGS.DISCOVER_DEPENDENCIES = "sys"


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


def setup(ex):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # Define configurations
    @ex.config
    def configurations():
        # logging
        observer = "mongo"              # str, which observer to use, [none|mongo|file_storage]
        if observer == "mongo":
            host = "localhost"          # str, MongoDB host address
            port = 7000                 # int, MongoDB port
        tag = "default"                 # str, Configuration tag
        logdir = "logs"                 # str, Directory to save checkpoint and logs

        # model
        model = "din"                   # str, model name, [unet2d|unet3d|din]
        if model == "din":
            init_channel = 30           # int, initialize number of channels of DIN
            max_channels = 320          # int, maximum number of channels of DIN
        normalizer = "in"               # str, Normalizer. [in|bn]
        normalizer_params = {
            "center": True,
            "scale": True,
            "epsilon": 1e-3
        }
        n_class = 2                     # str, number classes

        # training
        tf_func = True                  # bool, Enable tf.function for better performance
        epochs = 10                     # int, Number of epochs for training
        lr = 3e-4                       # float, Base learning rate for model training
        lrp = "period"                  # str, Learning rate policy [custom|period|poly|plateau]
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
        clipnorm = None
        loss_w_type = "none"            # str, class weights [none|numerical]
        loss_w = (1, 3)                 # list, loss weight, one value per class

        # data loader
        fold = 0                        # int, Test fold number, other folds are training folds
        depth = 10                      # int, Volume depth
        height = 512                    # int, Volume height
        width = 160                     # int, Volume width
        channel = 1                     # int, Volume channel
        bs = 8                          # int, Batch size
        zoom = (1.0, 1.25)              # tuple, Zoom range
        rotate = 5                      # float, rotate angle range
        flip = 7                        # int, Random flip flag
        tumor_percent = 0.5             # float, Tumor percentage
        guide = "exp"                   # str, Guide type [none|exp|euc|geo]
        exp_stddev = (1., 5., 5.)       # float, Stddev for exponential distance guide
        train_n = 1600                  # int, Number of samples per training epoch
        val_n = train_n // 10           # int, Number of samples for evaluation
        test_depth = -1                 # int, test image depth
        test_height = 960               # int, test image height
        test_width = 320                # int, test image width
        num_inters_max = 11             # int, Maximum number of interactions
        geo_lamb = 1.0                  # float, geodesic lambda, 1.0 for geo, 0.0 for euc
        geo_iter = 1                    # int, geodesic iteration
        use_box = False                 # bool, Use pre-defined boxes when evaluating
        test_set = "eval"               # str, test dataset [eval|test|extra]

        # testing
        resume_dir = ""                 # str, resume checkpoint from this directory
        save_dir = ""                   # str, directory to save prediction
        metrics = ["Dice", "VOE", "RVD"]    # list, Test metric names. [Dice/VOE/RVD/ASSD/RMSD/MSD]
        tta = True                      # bool, Enable test time augmentation (flip)

        log_step = 500              # int, Log running information per `log_step`
        summary_step = log_step     # int, Summary user-defined items to event file per `summary_step`
        summary_prefix = tag        # str, A string that will be prepend to the summary tags
        ckpt_freq = 1               # int, Frequency to take snapshot of the model weights. 0 denotes no snapshots.
        ckpt_best = True            # bool, Keep best snapshot or not

        enable_tensorboard = True       # bool, Enable Tensorboard or not
        save_best_ckpt = True           # bool, Save best checkpoint or not
        save_best_interval = (200,)     # tuple, Save best checkpoint every interval
        export_id = 0                   # int, export model with an id


        """ ==> Device Arguments """
        device_mem_frac = 0.            # Used for per_process_gpu_memory_fraction
        distribution_strategy = "off"   # === Don't use! === A string specify which distribution strategy to use
        gpus = 0                        # Which gpu to run this model. For multiple gpus: gpus=[0, 1]
        all_reduce_alg = ""             # === Don't use! === Specify which algorithm to use when performing all-reduce

    @ex.config_hook
    def config_hook(config, command_name, logger):
        # Create observers
        add_observers(ex, config, db_name=ex.path)
        ex.logger = loggers.get_global_logger(name=ex.path)

        # Limit gpu memory usage
        all_gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in all_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return {}

    return ex


def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    tf.random.set_seed(opt.seed)


def initialize(ex, run, config):
    opt = MapConfig(config)
    set_seed(opt)
    ex.logger.info(f"Run: {' '.join(sys.argv)}")

    logdir = Path(opt.logdir) / opt.tag / str(run._id)
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
    ex.logger.info(f"Log: {logdir.absolute()}")
    run.logdir_ = str(logdir)

    return opt, ex.logger
