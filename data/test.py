# _*__coding:utf-8 _*__
# @Time :2021/7/13 16:22
# @Author :bay
# @File test.py
# @Software : PyCharm
import jieba
import os


def readfiles():
    path = r'D:\workspace\pycharm\RL\LSTM0627\第一步\fake_content'  # 文件夹目录
    files = os.listdir(path)    # 得到文件夹下的所有文件名称
    txts_conserve = []  # 用于保存读出文档，并按 读文档名的顺序排列
    file_names = []  # 用于保存文件名的数组，方便后期匹配查找
    for file in files[:2]:  # 遍历文件夹
        position = path+'\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
        file_names.append(file)
        with open(position, "r", encoding='utf-8') as f:    # 打开文件
            data = f.read()   # 读取文件
            txts_conserve.append(data)
    return txts_conserve, file_names


def process_files(passage_num=6):
    row_txts, txts_names = readfiles()  # 得到两个数组， 一个为 用户名 ， 一个为 原生文档
    processed_files = []   # 用于保存分词后每个用户的结果
    processed_users = [[] for i in range(len(row_txts))]   # 将单个用户的信息进行保存，以倒叙读取，篇数取自于 passage_num
    stopwords = []   # 停词库
    with open('D:\workspace\pycharm\RL\LSTM0627\第一步\stop_words.txt', 'r', encoding="utf-8") as f:    # 读取停词库
        lines = f.readlines()
        f.close()
    for l in lines:
        stopwords.append(l.strip())

    def seg_sentence(sentence):
        print("sentence", sentence)
        key = 0  # 统计该用户发布信息条数
        value = []  # 对应每条信息发布的内容
        row_sentence = sentence.replace('，', '').replace('、', '').replace('？', ''). \
            replace('//', '').replace('/', '').replace('NULL', '').lstrip()  # 对句子进行预处理
        # print("row_sentence", row_sentence)
        sentence_seged = jieba.cut(row_sentence)  # 对句子进行分词
        # print("sentence_seged", sentence_seged)
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

    for i in range(len(row_txts)):  # 读取每一个用户的文档
        line_segs = seg_sentence(row_txts[i])  # 这里为list
        print("line_segs", line_segs)


if __name__ == '__main__':
    process_files(passage_num=6)
