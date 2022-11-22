import os
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
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
#from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import csv
import codecs
data_home = 'new_data'
dataset_name = 'new_data'



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
#    self.nb_words = self.data['nb_words'] # len of word dic
##    print(self.data.keys())
#    self.train = self.data['train']
#    self.vali = self.data['vali']
#    self.test = self.data['test']
#    self.nb_train = self.data['nb_train']
#    self.nb_vali = self.data['nb_vali']
#    self.nb_test = self.data['nb_test']
#    self.nb_users = self.data['nb_users']
    # 聚类所用的input (nb_users, seqlen, maxlen)
#    self.cls_x = self.get_cluster_x()

  def get_cluster_x(self):
    users = self.data['user']
    print("users", users)
#    users和texts指编号
    texts = self.data['texts']
    print("texts", texts[:10])
    texts = BasicData.pad_text(texts, self.maxlen)
    print("texts", texts[:10])
    padding = BasicData.pad_text([[1]], self.maxlen)[0]
    print("padding", padding)

    # print('#users: ', self.nb_users)
    x = [[] for i in range(len(users))]
    for i in range(len(texts)):
      if len(x[users[i]]) < self.seqlen:
        x[users[i]].append(texts[i])
    for i in range(len(users)):
      while len(x[i]) < self.seqlen:
        x[i].append(padding)
#    print('x',x[:10])
    return x

  @staticmethod
  def pad_text(texts, maxlen):
    text = pad_sequences(texts, maxlen = maxlen, padding = 'post', truncating = 'post')
    return text

#####################################
# PCA降维+Kmeans聚类
#####################################
#def PCAandKmeans():
#  X = np.array(cls_x)
#  X = X.reshape(X.shape[0],-1)
#  print(X.shape)
#  # std = StandardScaler()
#  # X = std.fit_transform(X)
#
#  X_scaled = preprocessing.scale(X)
#  from sklearn.decomposition import PCA
#  pca = PCA(n_components=3) # 降成3维
#  X_pca = pca.fit_transform(X_scaled)
#
#  s_score = []
#  d = {}
#  for i in range(2, 14):
#    y_pred = KMeans(n_clusters=i).fit_predict(X_pca)
#    tmp_score = metrics.silhouette_score(X_pca, y_pred)
#    # calinski_harabaz_score = metrics.calinski_harabasz_score(X_pca, y_pred)
#    d.update({i: tmp_score})
#    print('calinski_harabaz_score with i={0} is {1}'.format(i, tmp_score))
#    ax = plt.subplot(4, 3, i - 1, projection='3d')
#    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y_pred)

#PCAandKmeans()



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

#    keras.utils.plot_model(encoder, show_shapes=True)
#    plotting
    encoded_texts = encoder.predict(self.X)
#    f = open("D:/spyder/111.txt", ‘w’, encoding=utf-8) 
#    f.write(encoded_texts)
#    f.close()
    
#    print ("encoded_texts:", encoded_texts.shape)
#    print('encoded_texts:', encoded_texts[:2])
    pr = tf.reduce_mean(encoded_texts)
    pr = 1*pr
    ap = AffinityPropagation(preference=-20, damping=0.8).fit(encoded_texts)
    cluster_centers_ = ap.cluster_centers_
    print ("cluster_centers_:", cluster_centers_)
    file_csv = codecs.open('./new_data/result.csv', 'w+', 'utf-8')
    writer = csv.writer(file_csv)
    for data in cluster_centers_:
      writer.writerow(data)
    file_csv.close()
    print("保存成功， 处理结束")

    cluster_centers_indices = ap.cluster_centers_indices_    # 预测出的中心点的索引，如[123,23,34]
    print ("cluster_centers_indices:", cluster_centers_indices)
    n_clusters_ = len(cluster_centers_indices)
    print ("n_clusters_:", n_clusters_)
    return autoencoder, encoder, n_clusters_

#BD = BasicData(data_home, 20, 5)
#cls_x = BD.get_cluster_x()
#PK = PCAandKmeans()

nb_clusters = 8
dim_k = 5
seqlen = 3
maxlen = 5
epochs = 1
update = 1
dataname = 'new_data'
cls_x = BasicData(dataset_name, maxlen, seqlen).get_cluster_x()
CL = ClusterLSTM(nb_clusters, cls_x, dim_k, seqlen, maxlen, epochs, update, dataname)
autoencoder, encoder, n_clusters_ = CL.get_cls_centers()
