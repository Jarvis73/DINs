from tensorflow.keras import optimizers, callbacks


class Solver(object):
    def __init__(self, opt):
        self.lr_callback = self._get_callback(opt)
        self.optim = self._get_model_optimizer(opt)

    @staticmethod
    def _get_callback(opt):
        if opt.lrp == "plateau":
            callback = callbacks.ReduceLROnPlateau(
                monitor=opt.monitor, factor=opt.lr_decay_rate, patience=opt.lr_patience,
                mode='min', min_delta=opt.lr_min_delta, cooldown=opt.cool_down, min_lr=opt.lr_end)
        elif opt.lrp == 'period':
            scheduler = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=opt.lr, decay_steps=opt.lr_decay_step,
                decay_rate=opt.lr_decay_rate, staircase=True)
            callback = callbacks.LearningRateScheduler(scheduler)
        elif opt.lrp == "custom":
            scheduler = optimizers.schedules.PiecewiseConstantDecay(
                boundaries=opt.lr_boundaries, values=opt.lr_values)
            callback = callbacks.LearningRateScheduler(scheduler)
        elif opt.lrp == 'poly':
            scheduler = optimizers.schedules.PolynomialDecay(
                initial_learning_rate=opt.lr, decay_steps=opt.epochs,
                end_learning_rate=opt.lr_end, power=opt.lr_power)
            callback = callbacks.LearningRateScheduler(scheduler)
        else:
            raise ValueError(f'`lrp` supports [custom|period|poly|plateau], got {opt.lrp}')

        return callback

    @staticmethod
    def _get_model_optimizer(opt):
        kwargs = {} if opt.clipnorm is None else {"clipnorm": opt.clipnorm}
        if opt.optimizer == "adam":
            optimizer_params = {"beta_1": opt.beta_1, "beta_2": opt.beta_2, "epsilon": opt.epsilon}
            optimizer_params.update(kwargs)
            optimizer = optimizers.Adam(opt.lr, **optimizer_params)
        elif opt.optimizer == "momentum":
            optimizer_params = {"momentum": opt.momentum, "nesterov": opt.nesterov}
            optimizer_params.update(kwargs)
            optimizer = optimizers.SGD(opt.lr, **optimizer_params)
        else:
            raise ValueError(f"`optimizer` supports [adam/momentum], got {opt.optimizer}")

        return optimizer
