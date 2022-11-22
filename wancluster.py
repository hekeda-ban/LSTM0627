#import os
#os.chdir('/content/drive/My Drive/SeqCom/') #os.chdir() 方法用于改变当前工作目录到指定的路径。
import tensorflow as tf
import numpy as np, sys, math, random, json, time
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Activation, Embedding, Flatten, RepeatVector
from keras.layers import Lambda, Reshape, BatchNormalization, Dropout, TimeDistributed
from keras.layers import LSTM, Bidirectional, concatenate, add, multiply
from keras.models import Model
from keras import regularizers, optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
#####################################
# Data structure
#####################################
class BasicData:
  def __init__(self, dataset_name, maxlen, seqlen):
    '''
    dataset_name: /
    maxlen: text向量的维度，padding后
    seqlen: 用户历史post个数 
    '''
    self.dataset_name = dataset_name
    fn = '{}/{}.json'.format(data_home, dataset_name) #?
    self.data = json.load(open(fn, 'r'))
    self.maxlen = maxlen
    self.seqlen = seqlen
    self.nb_words = self.data['nb_words'] # len of word dic
#    print(self.data.keys())
    self.train = self.data['train']
    self.vali = self.data['vali']
    self.test = self.data['test']
    self.nb_train = self.data['nb_train']
    self.nb_vali = self.data['nb_vali']
    self.nb_test = self.data['nb_test']
    self.nb_users = self.data['nb_users']
    # 聚类所用的input (nb_users, seqlen, maxlen)
    self.cls_x = self.get_cluster_x()

  def get_cluster_x(self):
    users = self.train['user'] + self.vali['user'] + self.test['user']
#    print("users", users)
#    users和texts指编号
    texts = self.train['text'] + self.vali['text'] + self.test['text']
    print("texts", texts[:10])
    texts = BasicData.pad_text(texts, self.maxlen)
    #print("texts", texts[:10])
    padding = BasicData.pad_text([[1]], self.maxlen)
    #print("padding", padding)

    # print('#users: ', self.nb_users)
    x = [[] for i in range(self.nb_users)]
    for i in range(len(texts)):
      if len(x[users[i]]) < self.seqlen:
        x[users[i]].append(texts[i])
    for i in range(self.nb_users):
      while len(x[i]) < self.seqlen:
        x[i].append(padding)
    print('x',x[:10])
    return x

  @staticmethod
  def pad_text(texts, maxlen):
    text = pad_sequences(texts, maxlen = maxlen, padding = 'post', truncating = 'post')
    return text

#class KerasData:
#  # used for train/vali/test data
#  def __init__(self, data, batch_size, maxlen, cls_centers):
#    self.data = pd.DataFrame({
#        'user': data['user'],
#        'text': data['text'],
#        'label': data['label']
#    })
#    self.len = len(self.data)
#    # print('len: ',self.len)
#    self.batch_size = batch_size
#    self.maxlen = maxlen
#    self.cls_centers = cls_centers
#
#  def user(self, batch):
#    u = batch.user.values
#    u = u.reshape((-1, 1))
#    return u
#  
#  def text(self, batch):
#    t = batch.text.values
#    t = BasicData.pad_text(t, self.maxlen)
#    return t
#
#  def get_batch(self, batch):
#    # user: (?, 1) text: (?, maxlen)
#    x = []
#    x.append(self.user(batch))
#    x.append(self.text(batch))
#    # cls_centers: (nb_clusters, cls_centers_dim)
#    # tmp: (?, nb_clusters, cls_centers_dim)
#    # tmp_cls = self.cls_centers.tolist()
#    tmp = []
#    for i in range(len(batch)):
#      tmp.append(self.cls_centers)
#    x.append(np.array(tmp))
#    y = batch.label.values
#    y = np.log(y + 1).tolist()
#    return (x, y)
  
#  def generator(self, shuffle = False):
#    while True:
#      idx = list(range(self.len))
#      if shuffle:
#        np.random.shuffle(idx)
#      for i in range(0, self.len, self.batch_size):
#        batch = self.data.iloc[idx[i: i + self.batch_size]]
#        yield self.get_batch(batch)


#####################################
# LSTM Autoencoder学习用户历史序列偏好特征
#####################################
class ClusterLSTM:
  def __init__(self, nb_clusters, cls_x, dim_k, seqlen, maxlen, epochs, update, dataname):
    # cls_x (?, seqlen, maxlen)
    self.nb_clusters = nb_clusters
    self.X = np.array(cls_x)#(?, seqlen, maxlen)
    self.dim_k = dim_k
    self.seqlen = seqlen
    self.maxlen = maxlen
    self.epochs = epochs
    self.update = update
    self.dataname = dataname
    
    self.X = preprocessing.scale(self.X.reshape(-1, self.maxlen))
    self.X = self.X.reshape(-1, self.seqlen, self.maxlen)
    # print(self.X.shape)

  def get_cls_centers(self):
    if self.update == 0:
      encoder = keras.models.load_model(str(self.dataname)+'_encoder.h5', compile = False)
      autoencoder = keras.models.load_model(str(self.dataname)+'_autoencoder.h5', compile = False)
    else:
      input_texts = Input(shape=(self.X.shape[1], self.X.shape[2]), name='text_input')#Self.X数据即用户的历史行为序列数据，其维度为（None, seqlen, maxlen），分别表示（batch size，用户历史行为数量，每个行为文本的最大长度）。shape[1]和shape[2]即后两个维度。
      encoded = LSTM(self.dim_k, activation='relu', name='LSTM')(input_texts)

      decoded = RepeatVector(self.X.shape[1])(encoded)
      decoded = LSTM(self.X.shape[2], activation='relu', return_sequences=True)(decoded)
      # decoded = TimeDistributed(Dense(self.X.shape[2]))；# return_sequences用于判断输出向量(true)或序列(flase)
      # encoder = Model(inputs=input_texts, outputs=encoded)
      autoencoder = Model(inputs=input_texts, outputs=decoded)
      #模型compile，指定loss function， optimizer， metrics
      autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
      # print(autoencoder.summary())
      autoencoder.fit(self.X, self.X, batch_size=64, epochs=self.epochs, verbose=1)
      encoder = Model(inputs = autoencoder.inputs, outputs = autoencoder.layers[1].output)

      autoencoder.save(str(self.dataname)+'_autoencoder.h5')
      encoder.save(str(self.dataname)+'_encoder.h5')

    # keras.utils.plot_model(encoder, show_shapes=True)
    # plotting
    encoded_texts = encoder.predict(self.X)
    # print(encoded_texts[0])
    return autoencoder, encoder

maxlen = 5
seqlen = 5
nb_clusters = 8
dim_k = 3
epochs = 1
update = 1
dataname = 'tweet'
dataset_name = 'tweet'
data_home = 'data'
cls_x = BasicData(dataset_name, maxlen, seqlen)
CL = ClusterLSTM(nb_clusters, cls_x, dim_k, seqlen, maxlen, epochs, update, dataname)