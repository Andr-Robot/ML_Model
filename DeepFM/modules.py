import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Dense, Layer


class FM(Layer):
    """FM Part"""
    def __init__(self, feature_length, w_reg=1e-6):
        """
        这个函数用于对所有独立的输入进行初始化。（独立的输入：特指和训练数据无关的输入）(这个函数仅被执行一次)
        feature_length: A scalar. The length of features.
        w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        """
        这个函数用于当你知道输入Tensor的shape后，完成其余的初始化。（即需要知道数据的shape,但不需要知道数据的具体值）
        (注意：这个函数仅在Call被第一次调用时执行)
        """
        self.w = self.add_weight(name='w',
                                 shape=[self.feature_length, 1],
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        这个函数就是用来前向计算的函数了
        inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        # first order
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order

class DNN(Layer):
    """DNN Part"""
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """
        hidden_units: A list like `[unit1, unit2,...,]`. List of hidden layer units's numbers
        activation: A string. Activation function
        dnn_dropout: A scalar. dropout number
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
