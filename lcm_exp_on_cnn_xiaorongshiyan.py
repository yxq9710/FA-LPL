import datetime
import sys
import pandas as pd
import numpy as np
from utils import *
from sklearn.utils import shuffle
from models import cnn
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# ========== parameters & dataset: ==========
vocab_size = 20000
wvdim = 64
hidden_size = 64
num_filters = 100
filter_sizes = [3,10,25] # 不能超过maxlen
alpha = 4 # new model的loss中的alpha
batch_size = 128
epochs = 40
emb_type = ''

log_txt_name = '20NG_CNN_log'  # maxlen = 300
num_classes = 20
df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
df = pd.concat([df1,df2])
maxlen = 300
epochs = 20

df = df.dropna(axis=0,how='any')
df = shuffle(df)[:50000]
print('data size:',len(df))


# ========== data pre-processing: ==========
labels = sorted(list(set(df.label)))
assert len(labels) == num_classes,'wrong num of classes'
label2idx = {name:i for name,i in zip(labels,range(num_classes))}
num_classes = len(label2idx)

corpus = []
X_words = []
Y = []
i = 0
lens = []
for content,y in zip(list(df.content),list(df.label)):
    i += 1
    if i%1000 == 0:
        print(i)
    # English:
    content_words = content.split(' ')
    # Chinese:
    # content_words = jieba.lcut(content)

    corpus += content_words
    X_words.append(content_words)
    Y.append(label2idx[y])

tokenizer, word_index, freq_word_index = fit_corpus(corpus,vocab_size=vocab_size)
X = text2idx(tokenizer,X_words,maxlen=maxlen)
y = np.array(Y)


# ========== model traing: ==========
lcm_loss = []   # loss1
lcm_loss1 = []  # lcm + loss1
lcm_loss2 = []
N = 10
for n in range(N):
    # randomly split train and test each time:
    np.random.seed(n) # 这样保证了每次试验的seed一致
    random_indexs = np.random.permutation(range(len(X)))
    train_size = int(len(X)*0.6)
    val_size = int(len(X)*0.15)
    X_train = X[random_indexs][:train_size]
    X_val = X[random_indexs][train_size:train_size+val_size]
    X_test = X[random_indexs][train_size+val_size:]
    y_train = y[random_indexs][:train_size]
    y_val = y[random_indexs][train_size:train_size+val_size]
    y_test = y[random_indexs][train_size+val_size:]
    data_package = [X_train,y_train,X_val,y_val,X_test,y_test]


    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(str(datetime.datetime.now()),file=f)
        print('--Round',n+1,file=f)
    print('--Round ', n+1)
    # print("====loss1:==========")
    # for alpha in [3]:
    #     loss_lcm = cnn.CNN_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None, filter_sizes=filter_sizes, num_filters=num_filters)
    #     best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=10)
    #     lcm_loss.append(final_test_score)
    #     with open('output/%s.txt'%log_txt_name, 'a') as f:
    #         print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)

    print("====lcm loss no distill:==========")
    for alpha in [1]:
        loss_lcm = cnn.CNN_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None, filter_sizes=filter_sizes, num_filters=num_filters)
        best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=10)
        lcm_loss1.append(final_test_score)
        with open('output/%s.txt'%log_txt_name, 'a') as f:
            print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)

    print("====lcm loss no attention:==========")
    for alpha in [3]:
        loss_lcm = cnn.CNN_two_loss_no_attention(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None, filter_sizes=filter_sizes, num_filters=num_filters)
        best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=10)
        lcm_loss2.append(final_test_score)
        with open('output/%s.txt'%log_txt_name, 'a') as f:
            print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)

# print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)))
print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)))
print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss2), np.std(lcm_loss2)))
with open('output/%s.txt'%log_txt_name, 'a') as f:
    # print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)), file=f)
    print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)), file=f)
    print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss2), np.std(lcm_loss2)), file=f)
