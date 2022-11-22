import jieba
import os
from shutil import copy, rmtree
from gensim import corpora, models
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np


def readfiles():
    path = r"D:\workspace\pycharm\RL\LSTM0627\第一步\fake_content"  # 文件夹目录
    files = os.listdir(path)    # 得到文件夹下的所有文件名称
    txts_conserve = []  # 用于保存读出文档，并按 读文档名的顺序排列
    file_names = []  # 用于保存文件名的数组，方便后期匹配查找
    for file in files:  # 遍历文件夹
        position = path+'\\' + file # 构造绝对路径，"\\"，其中一个'\'为转义符
        file_names.append(file)
        print('position', position)
        with open(position, "r", encoding='utf-8') as f:    # 打开文件
            data = f.read()   # 读取文件
            txts_conserve.append(data)
            print('txts_conserve', txts_conserve)
    return txts_conserve, file_names


#  用于处理数据，剔除无效词
def process_files(passage_num=10):  #选取前10条信息
    row_txts, txts_names = readfiles()  # 得到两个数组， 一个为 用户名 ， 一个为 原生文档
    processed_files = []   # 用于保存分词后每个用户的结果
    processed_users = [[]for i in range(len(row_txts))]   # 将单个用户的信息进行保存，以倒叙读取，篇数取自于 passage_num
    stopwords = []   # 停词库

    with open('D:\workspace\pycharm\RL\LSTM0627\第一步\stop_words.txt', 'r', encoding="utf-8") as f:    # 读取停词库
        lines = f.readlines()
        f.close()

    for l in lines:
        stopwords.append(l.strip())

    def seg_sentence(sentence):
        key = 0  # 统计该用户发布信息条数
        value = []  # 对应每条信息发布的内容
        row_sentence = sentence.replace('，', '').replace('、', '').replace('？', ''). \
            replace('//', '').replace('/', '').replace('NULL', '').lstrip()  # 对句子进行预处理  lstrip() 方法用于截掉字符串左边的空格或指定字符
#        print('row_sentence:', row_sentence)
        sentence_seged = jieba.cut(row_sentence)  # 对句子进行分词
#        print('sentence_seged:', sentence_seged)
        outstr = ''
        for word in sentence_seged:  # 读取每一个单词
            if word == '=':
                if outstr != '':
                    value.append(outstr)
                    key += 1
                outstr = ''
                continue
            if word not in stopwords:
                if len(word) <= 5 and len(word) >= 1:
                    outstr += word
                    outstr += " "

        return value



    # line_segs = seg_sentence(row_txts[3])
    # for i in range(len(line_segs)):
    #     print(line_segs[i], '*************')
    for i in range(len(row_txts)):  # 读取每一个用户的文档
        line_segs = seg_sentence(row_txts[i]) # 这里为list
        print('line_segs', line_segs)
        print('line_segs的长度', len(line_segs))
        # 如果篇幅大于passage_num,则取前十篇
        if len(line_segs) > passage_num:
            for j in range(len(line_segs) - 1, len(line_segs) - 1 - passage_num, -1):
                processed_users[i].append(line_segs[j])
        # 否则全部取出
        else:
            for j in range(len(line_segs) - 1, -1, -1):
                processed_users[i].append(line_segs[j])
        # result = ''.join(line_segs)
        processed_files.append(processed_users[i])

    def mk_file(file_path: str):
        if os.path.exists(file_path):
            # 如果文件夹存在，则先删除原文件夹在重新创建
            rmtree(file_path)
        os.makedirs(file_path)
        
    # 保存清洗数据的结果
    relative_path = r'D:\workspace\pycharm\RL\LSTM0627\第一步\clean_data'
    mk_file(relative_path)
    for i in range(len(txts_names)):
        conserve_path = relative_path + '\\' + txts_names[i]
        with open(conserve_path, 'w', encoding='utf-8') as f:
            f.write('[')
            for j in range(len(processed_files[i])):
                f.write('{')
                f.write(str(processed_files[i][j]))
                f.write('},')
            f.write(']')
        f.close()

    return processed_files, txts_names

#def create_embedding(padded):
#    model = Sequential()
#    model.add(Embedding(935, 64, input_length=5))
##    model.add(LSTM(output_dim=128, return_sequences = True))
#    model.compile('rmsprop', 'mse')
#    embedding_matrix = model.predict(padded)
#    print("\nembedding_matrix:", embedding_matrix)
#    print("\nembedding_matrix:", embedding_matrix.shape)
#    return embedding_matrix

#def text_LSTM():
#    model = Sequential()
#    model.add(Embedding(len(words)+1, 256, input_length=maxlen))
#    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))

#  def text_emb(self):
#    x = Embedding(
#        input_dim = self.nb_words + 1,
#        output_dim = self.dim_k,
#        mask_zero = True,
#        name = 'text_embedding'
#    )(self.input_t) # (?, maxlen, dim_k)
#    x = LSTM(self.dim_k, return_sequences = True)(x) # (?, maxlen, dim_k)
#    return x


def text_emb(nb_words, dim_k):
   x = Embedding(
    input_dim=nb_words + 1,
    output_dim=dim_k,
    mask_zero=True,
    name='text_embedding'
    )(padded)   # (?, maxlen, dim_k)
   x = LSTM(dim_k, return_sequences=True)(x)    # (?, maxlen, dim_k)
   print('x:', x)
   return x


if __name__ == '__main__':
    files, names = process_files()
    sentences_whole = [[]for i in range(len(files))]
    for i in range(len(files)):
        for j in range(len(files[i])):
            sentences_whole[i].append(files[i][j])
        tokenizer = Tokenizer(num_words=935)  # OOV表示不在词典里的单词
        tokenizer.fit_on_texts(sentences_whole[i])
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(sentences_whole[i])
        padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)
#        padded = tf.convert_to_tensor(padded)
        # padding表示补齐0在末尾;truncating表示从句子的末尾开始舍弃
        print("对应文本：", names[i])
        print("\nWord Index=", word_index)
        print("\nSequences=", sequences)
        print("\npadded:", padded)
        text_emb(935,3)

        