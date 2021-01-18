import tensorflow as tf

def conv2d(x, filters, kernel, strides=1, dilation=1, scope=None, activation=None, reuse=None):
    with tf.variable_scope(scope):
        out=tf.layers.conv2d(x,filters,kernel,strides,padding='SAME', dilation_rate=dilation, kernel_initializer=tf.variance_scaling_initializer(), name='conv2d', reuse=reuse)

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)
