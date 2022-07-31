import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend


def tf_relu(x):
    return tf.nn.relu(x)


def tf_hard_sigmoid(x):
    return tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def tf_hard_swish(x):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _tf_se_block(inputs, filters, se_ratio, prefix):
    x = tf.reduce_mean(inputs, axis=[1, 2], name=prefix + 'squeeze_excite/AvgPool')
    x = tf.reshape(x, (-1, 1, 1, filters))
    x = tf.layers.conv2d(x,
                         _depth(filters * se_ratio),
                         kernel_size=1,
                         padding='same',
                         name=prefix + 'squeeze_excite/Conv')
    x = tf.nn.relu(x, name=prefix + 'squeeze_excite/Relu')
    x = tf.layers.conv2d(x,
                         filters,
                         kernel_size=1,
                         padding='same',
                         name=prefix + 'squeeze_excite/Conv_1')
    x = tf_hard_sigmoid(x)
    x = tf.multiply(inputs, x, name=prefix + 'squeeze_excite/Mul')
    return x


def _tf_inverted_res_block(x, expansion, filters, kernel_size, stride,
                           se_ratio, activation, block_id, training):
    channel_axis = -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = x.shape.as_list()[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = tf.layers.conv2d(x,
                             _depth(infilters * expansion),
                             kernel_size=1,
                             padding='same',
                             use_bias=False,
                             name=prefix + 'expand')
        x = tf.layers.batch_normalization(x,
                                          axis=channel_axis,
                                          epsilon=1e-3,
                                          momentum=0.999,
                                          training=training,
                                          name=prefix + 'expand/BatchNorm')
        x = activation(x)

    if stride == 2:
        x = _fixed_padding(x, [kernel_size, kernel_size])
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = tf.layers.batch_normalization(x,
                                      axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      training=training,
                                      name=prefix + 'depthwise/BatchNorm')
    x = activation(x)

    if se_ratio:
        x = _tf_se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = tf.layers.conv2d(x,
                         filters,
                         kernel_size=1,
                         padding='same',
                         use_bias=False,
                         name=prefix + 'project')
    x = tf.layers.batch_normalization(x,
                                      axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      training=training,
                                      name=prefix + 'project/BatchNorm')

    if stride == 1 and infilters == filters:
        x = shortcut + x
    return x


def mobilenet_v3(stack_fn,
                 inputs,
                 minimalistic=False,
                 training=False):
    img_input = inputs

    channel_axis = -1

    if minimalistic:
        kernel = 3
        activation = tf_relu
        se_ratio = None
    else:
        kernel = 5
        activation = tf_hard_swish
        se_ratio = 0.25

    x = _fixed_padding(img_input, [3, 3])

    x = tf.layers.conv2d(x,
                         16,
                         kernel_size=3,
                         strides=(2, 2),
                         padding='valid',
                         use_bias=False,
                         name='Conv')

    x = tf.layers.batch_normalization(x,
                                      axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      training=training,
                                      name='Conv/BatchNorm')
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    x = tf.layers.conv2d(x,
                         last_conv_ch,
                         kernel_size=1,
                         padding='same',
                         use_bias=False,
                         name='Conv_1')

    x = tf.layers.batch_normalization(x,
                                      axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      training=training,
                                      name='Conv_1/BatchNorm')
    x = activation(x)
    x = tf.identity(x, name="backbone_output")
    return x


def mobilenet_v3_small(inputs,
                       alpha=1.0,
                       minimalistic=False,
                       training=False):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _tf_inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, tf_relu, 0, training)
        x = _tf_inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, tf_relu, 1, training)
        x = _tf_inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, tf_relu, 2, training)
        x = _tf_inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3, training)
        x = _tf_inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4, training)
        x = _tf_inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5, training)
        x = _tf_inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6, training)
        x = _tf_inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7, training)
        x = _tf_inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8, training)
        x = _tf_inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9, training)
        x = _tf_inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10, training)
        return x

    return mobilenet_v3(stack_fn,
                        inputs,
                        minimalistic,
                        training)


def mobilenet_v3_large(inputs,
                       alpha=1.0,
                       minimalistic=False,
                       training=False):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _tf_inverted_res_block(x, 1, depth(16), 3, 1, None, tf_relu, 0, training)
        x = _tf_inverted_res_block(x, 4, depth(24), 3, 2, None, tf_relu, 1, training)
        x = _tf_inverted_res_block(x, 3, depth(24), 3, 1, None, tf_relu, 2, training)
        x = _tf_inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, tf_relu, 3, training)
        x = _tf_inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, tf_relu, 4, training)
        x = _tf_inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, tf_relu, 5, training)
        x = _tf_inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6, training)
        x = _tf_inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7, training)
        x = _tf_inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8, training)
        x = _tf_inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9, training)
        x = _tf_inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10, training)
        x = _tf_inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11, training)
        x = _tf_inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio,
                                   activation, 12, training)
        x = _tf_inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                   activation, 13, training)
        x = _tf_inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio,
                                   activation, 14, training)
        return x

    return mobilenet_v3(stack_fn,
                        inputs,
                        minimalistic,
                        training)


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    b = mobilenet_v3_large(a)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        c = sess.run(b, feed_dict={a: np.zeros((8, 224, 224, 3), dtype=float)})
    print(c.shape)
