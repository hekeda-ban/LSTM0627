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
import matplotlib.pyplot as plt
# matplotlib inline
import pandas as pd


class BasicData:
    def __init__(self, dataset_name, maxlen, seqlen):
        #  maxlen: text向量的维度，padding后
        #  seqlen: 用户历史post个数
        self.dataset_name = dataset_name
        # fn是拼接后的数据文件路径
        fn = '{}/{}.json'.format(data_home, dataset_name)
        self.data = json.load(open(fn, 'r'))
        self.maxlen = maxlen
        self.seqlen = seqlen
        # self.cls_x = self.get_cluster_x()

    def get_cluster_x(self):
        users = self.data['user']
        print("users的长度: ", len(users), users)
        texts = self.data['texts']
        print("texts的长度:", len(texts), texts)
        texts = BasicData.pad_text(texts, self.maxlen)  # 取二维数组中每行元素中前maxlen数,不够maxlen,在后面补零
        padding = BasicData.pad_text([[1]], self.maxlen)[0]  # 一维数组[1,0,0,0,0]
        x = [[] for i in range(len(users))]  # 构造一个二维空数组
        for i in range(len(texts)):
            if len(x[users[i]]) < self.seqlen:
                x[users[i]].append(texts[i])
        for i in range(len(users)):
            while len(x[i]) < self.seqlen:
                x[i].append(padding)
        return x

    @staticmethod
    def pad_text(texts, maxlen):
        text = pad_sequences(texts, maxlen=maxlen, padding='post', truncating='post')
        return text


class KerasData:
    # used for train/vali/test data
    def __init__(self, data, batch_size, maxlen, cls_centers):
        self.data = pd.DataFrame({
          'user': data['user'],
          'text': data['text'],
          'label': data['label']
        })
        # print("KerasData中的data", self.data)
        self.len = len(self.data)
        # print('len: ', self.len)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.cls_centers = cls_centers

    def user(self, batch):
          u = batch.user.values
          u = u.reshape((-1, 1))
          return u

    def text(self, batch):
        t = batch.text.values
        t = BasicData.pad_text(t, self.maxlen)
        return t

    def get_batch(self, batch):
        print("batch是啥样的：\n", batch)
        # user: (?, 1) text: (?, maxlen)
        x = []
        x.append(self.user(batch))
        x.append(self.text(batch))
        # cls_centers: (nb_clusters, cls_centers_dim)
        # tmp: (?, nb_clusters, cls_centers_dim)
        # tmp_cls = self.cls_centers.tolist()
        tmp = []
        for i in range(len(batch)):
          tmp.append(self.cls_centers)
        x.append(np.array(tmp))
        y = batch.label.values
        y = np.log(y + 1).tolist()
        return (x, y)

    def generator(self, shuffle=False):
        while True:
            idx = list(range(self.len))
            if shuffle:
                np.random.shuffle(idx)
            for i in range(0, self.len, self.batch_size):
                batch = self.data.iloc[idx[i: i + self.batch_size]]
                yield self.get_batch(batch)


#####################################
# PCA降维+Kmeans聚类
#####################################
def PCAandKmeans():
  X = np.array(tweet_data.cls_x)
  X = X.reshape(X.shape[0], -1)
  print(X.shape)
  # std = StandardScaler()
  # X = std.fit_transform(X)
  X_scaled = preprocessing.scale(X)
  from sklearn.decomposition import PCA
  pca = PCA(n_components=3)  # 降成3维
  X_pca = pca.fit_transform(X_scaled)
  s_score = []
  d = {}
  for i in range(2, 14):
    y_pred = KMeans(n_clusters=i).fit_predict(X_pca)
    tmp_score = metrics.silhouette_score(X_pca, y_pred)
    # calinski_harabaz_score = metrics.calinski_harabasz_score(X_pca, y_pred)
    d.update({i: tmp_score})
    print('calinski_harabaz_score with i={0} is {1}'.format(i, tmp_score))
    ax = plt.subplot(4, 3, i - 1, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_pred)
# PCAandKmeans()


class ClusterLSTM:
    def __init__(self, nb_clusters, cls_x, dim_k, seqlen, maxlen, epochs, update, dataname):
        self.nb_clusters = nb_clusters
        self.X = np.array(cls_x)
        self.dim_k = dim_k
        self.seqlen = seqlen
        self.maxlen = maxlen
        self.epochs = epochs
        self.update = update
        self.dataname = dataname
        # 标准化
        self.X = preprocessing.scale(self.X.reshape(-1, self.maxlen))
        self.X = self.X.reshape(-1, self.seqlen, self.maxlen)

    def get_cls_centers(self):
        if self.update == 0:
            encoder = keras.models.load_model(str(self.dataname) + '_encoder.h5', compile=False)
            autoencoder = keras.models.load_model(str(self.dataname) + '_autoencoder.h5', compile=False)
        else:
            # 构造编码器
            # self.X.shape[1] = 10
            # self.X.shape[2] = 5
            input_texts = Input(shape=(self.X.shape[1], self.X.shape[2]), name='text_input')
            # units：输出维度
            # activation：激活函数，为预定义的激活函数名（参考激活函数）
            # self.dim_k表示输出层的维数
            encoded = LSTM(units=self.dim_k, activation='relu', name='LSTM')(input_texts)
            #  将输入重复self.X.shape[1]次数
            decoded = RepeatVector(self.X.shape[1])(encoded)
            #  return_sequences用于判断输出向量(true)或序列(flase)
            decoded = LSTM(self.X.shape[2], activation='relu', return_sequences=True)(decoded)
            # decoded = TimeDistributed(Dense(self.X.shape[2]))；# return_sequences用于判断输出向量(true)或序列(flase)
            # encoder = Model(inputs=input_texts, outputs=encoded)
            autoencoder = Model(inputs=input_texts, outputs=decoded)
            # 模型compile，指定loss function， optimizer， metrics
            autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
            # print(autoencoder.summary())
            autoencoder.fit(self.X, self.X, batch_size=64, epochs=self.epochs, verbose=1)
            encoder = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[1].output)
            autoencoder.save(str(self.dataname) + '_autoencoder.h5')
            encoder.save(str(self.dataname) + '_encoder.h5')
            encoded_texts = encoder.predict(self.X)
            f = open("/data/encoded_texts.txt", 'w', encoding='utf-8')
            f.write(encoded_texts)
            f.close()

        return autoencoder, encoder


if __name__ == '__main__':
    # dataset_name = 'tweet'
    dataset_name = 'new_data'
    data_home = 'data'
    seqlen = 5
    maxlen = 10
    cls_centers = 2
    nb_clusters = 2
    dim_k = 3
    cls_x = BasicData(dataset_name, maxlen, seqlen).get_cluster_x()
    print(cls_x)
    epochs = 1
    update = 1
    dataname = './data/result1'
    ClusterLSTM(nb_clusters, cls_x, dim_k, seqlen, maxlen, epochs, update, dataname).get_cls_centers()



