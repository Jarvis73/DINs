import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras.regularizers import l2
from tensorflow_addons import layers as La


def unfold(inp):
    return inp // 100, (inp % 100) // 10, inp % 10


class ConvBlock(L.Layer):
    def __init__(self, opt, filters, kernel, strides=111,
                 up=False, up_kernel=None, up_strides=None, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        kernel = unfold(kernel)
        strides = unfold(strides)

        regu = l2(opt.weight_decay)
        self.conv1 = L.Conv3D(
            filters, kernel, strides, "same", use_bias=False, kernel_regularizer=regu,
            name="conv1")
        self.norm1 = La.InstanceNormalization(name="norm1")
        self.conv2 = L.Conv3D(
            filters, kernel, (1, 1, 1), "same", use_bias=False, kernel_regularizer=regu,
            name="conv2")
        self.norm2 = La.InstanceNormalization(name="norm2")
        self.relu = L.ReLU()

        if up:
            up_kernel = unfold(up_kernel)
            up_strides = unfold(up_strides)

            self.cat = L.Concatenate()
            self.up = L.Conv3DTranspose(
                filters, up_kernel, up_strides, "same", use_bias=False, kernel_regularizer=regu,
                name="up")

    def call(self, x, training=True, **kwargs):
        if hasattr(self, "up"):
            y, x = x
            x = self.cat([y, self.up(x)])
        if isinstance(x, tuple):
            x, guide = x
            x = self.relu(self.norm1(self.conv1(x)) + guide)
            x = self.relu(self.norm2(self.conv2(x)))
        else:
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.relu(self.norm2(self.conv2(x)))
        return x


class DINs(Model):
    def __init__(self, opt, logger, name="model"):
        super(DINs, self).__init__(name=name)
        self.init_channel = opt.init_channel
        self.max_channels = opt.max_channels

        def get_channel(layer):
            return min(self.init_channel * 2 ** layer, self.max_channels)

        self.down1 = ConvBlock(opt, get_channel(0), 133, 111, name=f"{name}/down1")
        self.down2 = ConvBlock(opt, get_channel(1), 133, 122, name=f"{name}/down2")
        self.down3 = ConvBlock(opt, get_channel(2), 333, 122, name=f"{name}/down3")
        self.down4 = ConvBlock(opt, get_channel(3), 333, 122, name=f"{name}/down4")
        self.bridge = ConvBlock(opt, get_channel(4), 333, 222, name=f"{name}/bridge")
        self.up4 = ConvBlock(opt, get_channel(3), 333, 111, True, 222, 222, name=f"{name}/up4")
        self.up3 = ConvBlock(opt, get_channel(2), 333, 111, True, 122, 122, name=f"{name}/up3")
        self.up2 = ConvBlock(opt, get_channel(1), 133, 111, True, 122, 122, name=f"{name}/up2")
        self.up1 = ConvBlock(opt, get_channel(0), 133, 111, True, 122, 122, name=f"{name}/up1")

        regu = l2(opt.weight_decay)
        self.dim = Sequential(layers=[
            L.MaxPool3D((1, 8, 8), (1, 8, 8)),
            L.Conv3D(get_channel(4), (3, 3, 3), (2, 2, 2), "same", use_bias=True, kernel_regularizer=regu)
        ], name="dim")

        self.final = L.Conv3D(
            opt.n_class, (1, 1, 1), use_bias=True, kernel_regularizer=regu, name=f"{name}/final")

        logger.info(" " * 11 + f"==> Model {self.__class__.__name__} created")

    def call(self, x, training=True, **kwargs):
        image, guide = x
        x = tf.concat((image, guide), axis=-1)
        small_guide = self.dim(guide)

        x1 = self.down1(x, training)
        x2 = self.down2(x1, training)
        x3 = self.down3(x2, training)
        x4 = self.down4(x3, training)
        x = self.bridge((x4, small_guide), training)
        x = self.up4((x4, x), training)
        x = self.up3((x3, x), training)
        x = self.up2((x2, x), training)
        x = self.up1((x1, x), training)
        x = self.final(x)

        return x

    def get_config(self):
        return {
            "init_channel": self.init_channel,
            "max_channels": self.max_channels
        }
