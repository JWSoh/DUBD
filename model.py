from ops import *

class Denoiser(object):
    def __init__(self, x, sigma, name, reuse=False):
        self.input = x
        self.sigma= sigma
        self.name = name
        self.reuse = reuse

        self.noise_encoder()
        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.conv1 = conv2d(self.input, 64, [3, 3], scope='conv1', activation=None)
            self.head = self.conv1
            for idx in range(5):
                self.head = self.RIRblock(self.head, 5, 'RIRBlock' + repr(idx))

            self.conv2 = conv2d(self.head, 64, [3, 3], scope='conv2', activation=None)
            self.residual = tf.add(self.conv1, self.conv2)

            self.conv3= conv2d(self.residual, 3, [3, 3], scope='conv3', activation=None)

            self.output = tf.add(self.conv3, self.input)

        tf.add_to_collection('InNOut', self.input)
        tf.add_to_collection('InNOut', self.output)

    def RIRblock(self, x, num, scope):
        with tf.variable_scope(scope):
            head = x
            for idx in range(num):
                head = self.resblock(head, 'RBlock' + repr(idx))
            out = conv2d(head, 64, [3, 3], scope='conv_out')

            out = out*self.gamma + self.beta

        return tf.add(out, x)

    def resblock(self, x, scope):
        with tf.variable_scope(scope):
            net1 = conv2d(x, 64, [3, 3], dilation=1, scope='conv1', activation='ReLU')
            out = conv2d(net1, 64, [3, 3], dilation=1, scope='conv2', activation=None)

        return tf.add(out, x)

    def noise_encoder(self):
        with tf.variable_scope('Noise_ENC'):
            net = conv2d(self.sigma, 128, [1,1], scope='linear', activation= 'ReLU')

            self.gamma = conv2d(net, 64,[1,1], scope='gamma', activation =None)
            self.beta = conv2d(net, 64,[1,1], scope='beta', activation =None)


class Estimator(object):
    def __init__(self, x, name, reuse=False):
        self.input = x
        self.name = name
        self.reuse = reuse

        self.build_model()

    def build_model(self):
        print('Build Model {}'.format(self.name))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.net = conv2d(self.input, 64, [3, 3], strides=2, dilation=1, scope='conv1', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1,scope='conv2', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net = conv2d(self.net, 64, [3, 3],  strides=2, dilation=1, scope='conv3', activation=None)
            self.net = tf.nn.relu(self.net)

            self.net1 = conv2d(self.net, 64, [3, 3], strides=1,  dilation=1, scope='conv4', activation=None)
            self.net = tf.nn.relu(self.net1)

            self.net2 = conv2d(self.net, 64, [3, 3],  strides=1, dilation=1, scope='conv5', activation=None)
            self.net = tf.nn.relu(self.net2)

            self.net = conv2d(self.net, 3, [3, 3], dilation=1, scope='conv_out', activation=None)
            self.output=tf.image.resize_bilinear(self.net, tf.shape(self.input)[1:-1])

        tf.add_to_collection('InNOut', self.input)
        tf.add_to_collection('InNOut', self.output)