import datetime
import sys
import pandas as pd
import numpy as np
from utils import *
from sklearn.utils import shuffle
from models import lstm
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# ========== parameters & dataset: ==========
vocab_size = 20000
maxlen = 100
wvdim = 64
hidden_size = 64
alpha = 4 # new model的loss中的alpha
batch_size = 128
epochs = 20
emb_type = ''

dataset_name = '20NG'
group_noise_rate = 0.3
df,num_classes,label_groups = load_dataset(dataset_name)
log_txt_name = '%s_lstm_log(group_noise=%s,comp+rec+talk)' % (dataset_name,group_noise_rate)

df = df.dropna(axis=0,how='any')
df = shuffle(df)[:50000]
print('data size:',len(df))

# ========== data pre-processing: ==========
labels = sorted(list(set(df.label)))

label2idx = {name:i for name,i in zip(labels,range(num_classes))}
num_classes = len(label2idx)

assert len(labels) == num_classes,'wrong num of classes'

corpus = []
X_words = []
Y = []
i = 0

m_len = 0
lens = []
contents = []
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

    # apply noise only on train set:  # 只对bert下的20NG有影响，对其它的均没有影响
    if group_noise_rate > 0:
        _, overall_noise_rate, y_train = create_asy_noise_labels(y_train, label_groups, label2idx, group_noise_rate)
        data_package = [X_train,y_train,X_val,y_val,X_test,y_test]
        with open('output/%s.txt' % log_txt_name, 'a') as f:
            print('-' * 30, '\nNOITCE: overall_noise_rate=%s' % round(overall_noise_rate, 2), file=f)

    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(str(datetime.datetime.now()),file=f)
        print('--Round',n+1,file=f)
    print('--Round ', n+1)
    # parameters = [1,3,5,10]
    parameters = [3]
    for alpha in parameters:
        print("====loss1:==========")
        loss_lcm = lstm.LSTM_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)
        best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=10)
        lcm_loss.append(final_test_score)
        with open('output/%s.txt'%(log_txt_name), 'a') as f:
            print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)
            print('val acc list:\n', str(val_score_list), '\n', file=f)

print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)))
with open('output/%s.txt'%log_txt_name, 'a') as f:
    print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)), file=f)
