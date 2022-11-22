from keras.layers import Layer
from keras import backend as K
# 文本特征、用户特征和圈层特征


class Avg(Layer):
    # 重写初始化方法
    def __init__(self, return_sequences=False, **kw):
        super().__init__(**kw)
        self.return_sequences = return_sequences
        print("self.return_sequences", self.return_sequences)

    # TODO x
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            x = x * K.expand_dims(mask, -1)
            x = K.sum(x, -2) / K.expand_dims(K.sum(mask, -1), -1)
            return x
        else:
            return K.mean(x, -2)

    # TODO c是什么，att是什么
    def att_output(self, c, att, mask=None):
        att = K.softmax(att)
        if mask is not None:
            att = att * K.cast(mask, 'float32')
            att = att / K.expand_dims(K.sum(att, -1), -1)
        self.att_value = att
        att_text = c * K.expand_dims(att, -1)
        if self.return_sequences:
            return att_text
        return K.sum(att_text, -2)

    # TODO inputs是什么
    def compute_mask(self, inputs, mask=None):
        if type(mask) == list:
            mask = mask[0]
        if self.return_sequences:
            return mask
        return None

    def compute_output_shape(self, inputs):
        if type(inputs) == list:
            inputs = inputs[0]
        if self.return_sequences:
            return inputs
        return (inputs[0], inputs[-1])


class Basic_add_W(Avg):
    def __init__(self, l2=None, **kw):
        self.l2 = l2
        super().__init__(**kw)

    def add_weight(self, name, shape, add_l2=False):
        if add_l2:
            return super().add_weight(name, shape, regularizer=self.l2, initializer='glorot_normal')
        else:
            return super().add_weight(name, shape, initializer='glorot_normal')

# 正态化的Glorot初始化——glorot_normal
# wh * (tanh(wt * t) + tanh(wa * a) + tanh(wb * b))


class Att6(Basic_add_W):
    #  TODO input_shape
    def build(self, input_shape):
        maxlen = input_shape[0][1]
        dim_k = input_shape[0][2]
        dim_h = dim_k
        self.wt = self.add_weight('wt', (dim_k, dim_h), add_l2=True)
        self.wa = self.add_weight('wa', (dim_k, dim_h), add_l2=True)
        self.wb = self.add_weight('wb', (dim_k, dim_h), add_l2=True)
        self.wh = self.add_weight('wh', (dim_h, 1), add_l2=True)
        super().build(input_shape)

    # TODO x
    def call(self, x, mask=None):
        # t: (None, maxlen, dim_k), u: (None, dim_k), mask: (None, maxlen)
        t, a, b = x
        maxlen = K.shape(t)[1]
        # (None, maxlen, dim_h)
        wt = K.tanh(K.dot(t, self.wt)) #dot()函数是矩阵乘
        # (None, dim_h)
        wa = K.tanh(K.dot(a, self.wa))
        wb = K.tanh(K.dot(b, self.wb))
        h = wt * K.expand_dims(wa * wb, 1)
        att = K.reshape(K.dot(h, self.wh), (-1, maxlen))
        #att = K.tanh(att)
        if type(mask) == list:
            mask = mask[0]
        return self.att_output(t, att, mask)


