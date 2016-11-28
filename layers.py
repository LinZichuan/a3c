import tensorflow as tf

def Conv2D(name, inputx, out_channel, kernel_shape, stride=1):
    init = tf.truncated_normal_initializer(0, 0.02)
    in_channel = inputx.get_shape()[-1]  ##NHWC!!!
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_shape, kernel_shape, in_channel, out_channel], initializer=init)
        b = tf.get_variable('b', [out_channel], initializer=init)
        #stride = [1,1,1,1]
        conv = tf.nn.conv2d(inputx, w, stride, 'VALID', data_format='NHWC')
        l = tf.nn.bias_add(conv, b, data_format='NHWC')
        l = tf.nn.relu(l)
        print l.get_shape()
        return l

def MaxPooling(name, inputx, pool_size):
    with tf.variable_scope(name):
        l = tf.nn.max_pool(inputx, [1,pool_size,pool_size,1], [1,pool_size,pool_size,1], 'VALID', 'NHWC', 'pool')
        print l.get_shape()
        return l

def Linear(name, inputx, out_dim, nl=tf.identity):
    init = tf.truncated_normal_initializer(0, 0.02)
    shape = inputx.get_shape().as_list()
    #print shape
    inputx = tf.reshape(inputx, [-1, reduce(lambda x,y: x*y, shape[1:])])
    shape = inputx.get_shape().as_list()
    #print shape
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], out_dim], initializer=init)
        b = tf.get_variable('b', [out_dim], initializer=init)
        l = tf.nn.bias_add(tf.matmul(inputx, w), b)
        l = nl(l)
        print l.get_shape()
        return l
