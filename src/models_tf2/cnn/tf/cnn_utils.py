import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    # Get the number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width,
                                                    input_channels // groups, num_filters], trainable=True, initializer=weight_initer)
        biases = tf.get_variable('biases', shape=[num_filters], trainable=True, initializer=weight_initer)

    if groups == 1:
        conv = convolve(x, weights)

        # In the case of multiple groups, split inputs & weights convolve them
        # separately. (e.g AlexNet)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    return tf.nn.bias_add(conv, biases, name=name)


def fc(x, num_in, num_out, name):
    # Create tf variables for the weights and biases
    weights = tf.get_variable(name+"_W", shape=[num_in, num_out], trainable=True, initializer=weight_initer)
    biases = tf.get_variable(name+"_b", shape=[num_out], trainable=True, initializer=weight_initer)

    # Matrix multiply weights and inputs and add bias
    return tf.nn.xw_plus_b(x, weights, biases, name=name)


def batch_norm_layer(x, name):
    return tf.layers.batch_normalization(inputs=x, axis=3, center=True,
          scale=False, training=False, fused=True,  name=name)


def max_pool(x, filter_height, fileter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, fileter_width, 1],
                          strides=[1, stride_y, stride_x, 1], padding=padding,
                          name=name)


def avg_pool(x, filter_height, fileter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, filter_height, fileter_width, 1],
                          strides=[1, stride_y, stride_x, 1], padding=padding,
                          name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
