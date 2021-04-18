import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops.gen_batch_ops import batch

keras.backend.set_floatx('float64')
dtype = tf.float64

class SparseWeightInitializer():
    def __init__(self, density=0.2, sr_scale=1.0):
        self._density = density
        self._sr_rate = sr_scale

    def __call__(self, shape, dtype=dtype):
        self.w_init = tf.random.normal(shape, dtype=dtype)
        self.mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), self._density), dtype=dtype)
        w_sparse = self.w_init * self.mask
        eigenvalue, _ = tf.linalg.eigh(w_sparse)
        spectral_radius = tf.math.reduce_max(tf.abs(eigenvalue))
        return self._sr_rate * (w_sparse / spectral_radius)

class ESNCell(keras.layers.Layer):
    def __init__(self, units=10, fb_mode=False, sr_scale=1.0, density=0.2, leaking_rate=0.9, fb_rate=1.0, noise_rate=0.0, inputs=1, outputs=1, name="ESNCell", dtype=dtype, **kwargs):
        super(ESNCell, self).__init__(name=name, dtype=dtype, **kwargs)
        self.fb_mode = fb_mode
        self._sr_scale = sr_scale
        self._density = density
        self._leaking_rate = leaking_rate
        self._w_initializer = SparseWeightInitializer(density=self._density, sr_scale=self._sr_scale)
        self._inputs = inputs
        self._outputs = outputs
        self._units = units
        self._fb_rate = fb_rate
        self._noise_rate = noise_rate
        self._rate = 1.0

        self.state_size = [tf.TensorShape(self._units,), tf.TensorShape(self._outputs)]
        self.output_size = [tf.TensorShape(self._units), tf.TensorShape(self._outputs)]

        self.w = keras.layers.Dense(    units=self._units, activation=None, use_bias=True, kernel_initializer=self._w_initializer,
                                        bias_initializer='zeros', trainable=False, dtype=dtype)

        self.w_in = keras.layers.Dense( units=self._units, activation=None, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(minval=-self._rate, maxval=self._rate),
                                        trainable=False, dtype=dtype)

        self.w_fb = keras.layers.Dense( units=self._units, activation=None, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(minval=-self._fb_rate, maxval=self._fb_rate),
                                        trainable=False, dtype=dtype)

        self.activation = keras.layers.Activation('tanh')

        self.w_out = keras.layers.Dense( units=self._outputs, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                         bias_initializer='zeros', trainable=True, dtype=dtype)


    def build(self, input_shape):
        print('ESNCell build!')
        self.w.build(input_shape=[None, self._units])
        self.w_in.build(input_shape=[None, input_shape[0][-1]])
        self.w_fb.build(input_shape=[None, self._outputs])
        self.w_out.build(input_shape=[None, self._units])
        self.activation.build(input_shape=[None, self._units])
        self.built = True

    def call(self, inputs, states, **kwargs):

        input_data, teacher_fb = tf.nest.flatten(inputs)
        x_n_1, y_n_1 = states # x[n-1], y[n-1]

        if self.fb_mode is True:
            fb = self.w_fb(y_n_1)
        else:
            fb = self.w_fb(teacher_fb)

        # x_n = self.activation(self.w_in(input_data) + self.w(x_n_1) + fb + self._noise_rate*tf.random.uniform(x_n_1.shape, dtype=dtype))
        x_n = self.activation(self.w_in(input_data) + self.w(x_n_1) + fb)
        x_n = (1 - self._leaking_rate)*x_n_1 + self._leaking_rate*x_n
        y_n = self.w_out(x_n)

        return [x_n, y_n], [x_n, y_n]

    def set_fb_mode(self, fb_mode=True):
        self.fb_mode = fb_mode

    def get_config(self):
        config = super(ESNCell, self).get_config()
        config.update({'units': self._units, 'inputs': self._inputs, 'outputs': self._outputs})
        return config

class ReservoirModel(keras.Model):
    def __init__(self, units=10, batch_size=1, fb_mode=False, sr_scale=1.0, density=0.2, leaking_rate=0.9, fb_rate=1.0, noise_rate=0.0, inputs=1, outputs=1, name="Reservoir", dtype=dtype, **kwargs):
        super(ReservoirModel, self).__init__(name=name,**kwargs)
        self._units = units
        self._batch_size = batch_size
        self._fb_mode = fb_mode
        self._sr_scale = sr_scale
        self._density = density
        self._leaking_rate = leaking_rate
        self._fb_rate = fb_rate
        self._noise_rate = noise_rate
        self._inputs = inputs
        self._outputs = outputs

        #  layer setting
        self.esn_cell = ESNCell(units=self._units, fb_mode=self._fb_mode, sr_scale=self._sr_scale, density=self._density, leaking_rate=self._leaking_rate, fb_rate=self._fb_rate,
                                noise_rate=self._noise_rate, inputs=self._inputs, outputs=self._outputs)

        self.esn_layer = keras.layers.RNN(self.esn_cell, return_sequences=True, stateful=True, name='RNN_layer')

    def call(self, inputs, training):

        input_data, feedback = inputs
        y = self.esn_layer((input_data, feedback))
        return y


def Tikhonov(x, teacher_data, beta=0.001):
    teacher_data = np.reshape(teacher_data,(-1,1))
    x_n = np.reshape(x, (-1, x.shape[-1]))
    one = tf.ones([x_n.shape[0], 1], dtype=dtype)
    X = tf.concat([x_n, one], axis=1)
    w_new = tf.linalg.pinv(tf.transpose(X) @ X + beta * tf.linalg.eye(X.shape[1], dtype=dtype)) @ tf.transpose(X) @ teacher_data

    return w_new

def generate_slide_window(data, window_size=10):
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size)])

if __name__ == "__main__":
    print(tf.__version__)
    data = np.loadtxt("./ReservoirComputing/EchoStateNetwork/DuffingOscillatorData.txt")

    learning_step = 300
    units = 300
    density = 0.2
    sr_scale = 1.2
    leaking_rate = 0.5
    fb_rate = 0.5
    window_size = 100
    batch_size = 1

    learning_data = data[ :learning_step]
    test_data = data[learning_step:]

    learning_data_window = generate_slide_window(learning_data, window_size=window_size)
    learning_teacher_data = learning_data[window_size:]
    test_data_window = generate_slide_window(test_data, window_size=window_size)

    input_data = learning_data_window.reshape([batch_size, -1, learning_data_window.shape[-1]])
    print(input_data.shape)

    feedback_data = learning_teacher_data.reshape([batch_size, -1, learning_teacher_data.shape[-1]])
    print(feedback_data.shape)

    model = ReservoirModel(units=units, density=density, sr_scale=sr_scale, leaking_rate=leaking_rate, fb_rate=fb_rate, inputs=input_data.shape[-1], outputs=1)
    model.esn_cell.set_fb_mode(True)

    dummy = np.zeros(input_data.shape)

    x, y = model([input_data, dummy])

    plt.figure()
    plt.plot(np.reshape(y,-1))
    plt.plot(learning_teacher_data)

    w_new = Tikhonov(x=x, teacher_data=learning_teacher_data)


    model.esn_layer.reset_states()
    model.esn_cell.w_out.kernel.assign(w_new[0: -1, :])
    model.esn_cell.w_out.bias.assign(w_new[-1])

    # model.esn_layer.reset_states()
    # x, y = model([input_data, dummy])
    # plt.figure()
    # plt.plot(np.reshape(y, -1))
    # plt.plot(learning_teacher_data)
    # plt.show()

    model.esn_layer.reset_states()
    xx = []
    yy = []
    recurrent_data = np.array(input_data[0][-1])
    for i in range(len(data)):
        if i == 0:
            x, y = model([input_data, dummy])

        else:
            x, y = model([recurrent_data, dummy[0][0].reshape(1,1,-1)])

        recurrent_data = np.append(recurrent_data, (np.reshape(y, -1)[-1]))
        recurrent_data = np.roll(recurrent_data, -1)[0:window_size]
        recurrent_data = recurrent_data.reshape(1, -1, window_size)
        yy = np.append(yy, np.reshape(y, -1))

    
    plt.figure()
    plt.plot(yy)
    plt.plot(data[window_size: ])
    plt.show()

